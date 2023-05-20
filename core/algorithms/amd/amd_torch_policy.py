"""
PyTorch policy class used for AMD.
"""
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from ray.rllib.algorithms.a3c.a3c_torch_policy import A3CTorchPolicy
from ray.rllib.evaluation import Episode
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.postprocessing import Postprocessing, compute_gae_for_sample_batch
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch import torch_modelv2
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.typing import TensorStructType, TensorType
from torch.func import functional_call, jacrev

from .constants import PreLearningProcessing
from .utils import action_to_reward, discounted_cumsum_factor_matrix


class AMDAgent:
    """common functions that will be used by both torch and tf policy"""

    def compute_awareness(self, sample_batch: SampleBatch):
        """Calculate advantage function from critic, and also calculate awareness w.r.t. the whole batch"""
        raise NotImplementedError

    def loss_agent(
        self,
        model: ModelV2,
        dist_class: ActionDistribution,
        train_batch: SampleBatch,
    ) -> Union[TensorType, List[TensorType]]:
        """Loss for agent"""
        raise NotImplementedError


class AMDPlanner:

    appearance: np.ndarray = None  # this variable indicates which individual agent appears in game, used for reward calculation

    def loss_central_planner(
        self,
        model: ModelV2,
        dist_class: ActionDistribution,
        train_batch: SampleBatch,
    ) -> Union[TensorType, List[TensorType]]:
        """Loss for agent"""
        raise NotImplementedError


class AMDGeneralPolicy(AMDPlanner, AMDAgent, Policy):

    is_central_planner: bool = False

    def loss(
        self,
        model: ModelV2,
        dist_class: ActionDistribution,
        train_batch: SampleBatch,
    ) -> Union[TensorType, List[TensorType]]:

        if self.is_central_planner:
            return self.loss_central_planner(model=model, dist_class=dist_class, train_batch=train_batch)
        else:
            return self.loss_agent(model=model, dist_class=dist_class, train_batch=train_batch)


class AMDAgentTorchPolicy(AMDGeneralPolicy, A3CTorchPolicy):
    """Pytorch policy class used for Adaptive Mechanism design."""

    model: nn.Module
    agent_info_space: spaces.Space  # used for agent to collect reward from central planner

    @override(A3CTorchPolicy)
    def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        if self.is_central_planner:
            return convert_to_numpy({
                "planner_policy_loss": torch.mean(torch.stack(self.get_tower_stats("planner_policy_loss"))),
                "planner_reward_cost": torch.mean(torch.stack(self.get_tower_stats("planner_reward_cost"))),
                "cur_lr": self.cur_lr,
            })
        else:
            return convert_to_numpy({
                "agent_policy_loss": torch.mean(torch.stack(self.get_tower_stats("agent_policy_loss"))),
                "agent_value_loss": torch.mean(torch.stack(self.get_tower_stats("agent_value_loss"))),
                "cur_lr": self.cur_lr,
            })

    def postprocess_trajectory(
        self,
        sample_batch: SampleBatch,
        other_agent_batches: Dict[Any, SampleBatch] | None = None,
        episode: Episode | None = None,
    ) -> SampleBatch:

        if not self.is_central_planner:
            # ! STEP 1: calculate gae, this will add ADVANTAGEs and VALUE_TAEGETS into this batch
            # central planner does not need critic
            sample_batch = compute_gae_for_sample_batch(self, sample_batch, other_agent_batches, episode)

            # ! STEP 2: placeholder for reward from planner, this is for skipping Policy._initialize_loss_from_dummy_batch().
            # Since the default class is agent, we only have to pass a r_planner term
            sample_batch[PreLearningProcessing.R_PLANNER_CUM] = 0 * sample_batch[SampleBatch.REWARDS]
            sample_batch[PreLearningProcessing.AWARENESS] = 0 * sample_batch[SampleBatch.REWARDS]

        return sample_batch

    def loss_agent(
        self,
        model: torch_modelv2,
        dist_class: ActionDistribution,
        train_batch: SampleBatch,
    ) -> TensorType | List[TensorType]:

        # Pass the training data through our model to get distribution parameters.
        dist_inputs, _ = model(train_batch)
        # get value estimation by critic
        values = model.value_function()

        # Create an action distribution object.
        action_dist = dist_class(dist_inputs, model)

        # Calculate the vanilla PG loss based on: L = -E[ log(pi(a|s)) * A]
        log_probs = action_dist.logp(train_batch[SampleBatch.ACTIONS])

        # the loss contains two parts, first the real rewards, by critit
        # and the second: the reward from central planner
        pg_r_real = -torch.sum(log_probs * train_batch[Postprocessing.ADVANTAGES])
        # ps: if this agent is not in cooperative list, its r_planner_cum is zero
        pg_r_planner = -torch.sum(log_probs * train_batch[PreLearningProcessing.R_PLANNER_CUM])

        # Final policy loss.
        policy_loss = pg_r_real + pg_r_planner

        # Compute a value function loss.
        if self.config["use_critic"]:
            value_loss = 0.5 * torch.sum(torch.pow(values.reshape(-1) - train_batch[Postprocessing.VALUE_TARGETS], 2.0))
        else:  # Ignore the value function.
            value_loss = 0.0

        # Store values for stats function in model (tower), such that for
        # multi-GPU, we do not override them during the parallel loss phase.
        model.tower_stats["agent_policy_loss"] = policy_loss
        model.tower_stats["agent_value_loss"] = value_loss

        total_loss = policy_loss + self.config['vf_loss_coeff'] * value_loss

        return total_loss

    def loss_central_planner(
        self,
        model: torch_modelv2,
        dist_class: ActionDistribution,
        train_batch: SampleBatch,
    ) -> TensorType | List[TensorType]:

        # Pass the training data through our model to get distribution parameters.
        dist_inputs, _ = model(train_batch)

        policy_loss = torch.zeros_like(dist_inputs, requires_grad=True).mean()
        reward_cost = torch.zeros_like(policy_loss, requires_grad=True)

        # only compute loss when reward_max is not zero, for saving computation
        if self.config['planner_reward_max'] > 0.0:

            # Create an action distribution object.
            action_dist = dist_class(dist_inputs, model)

            # Compute the actual actions
            actions = action_dist.deterministic_sample()

            # get reward from actions
            r_planner = action_to_reward(
                actions=actions,
                appearance=self.appearance,
                reward_max=self.config['planner_reward_max'],
                zero_sum=self.config['force_zero_sum'],
                availability=train_batch[PreLearningProcessing.AVAILABILITY],
            )

            # get the matrix for computing accumulated reward
            # fix: somehow on clusters, discoutned factor matrix is not T x T
            cp_t = train_batch[SampleBatch.T]
            cp_eps_id = train_batch[SampleBatch.EPS_ID]
            cumsum_factor_matrix = discounted_cumsum_factor_matrix(eps_id=cp_eps_id, t=cp_t)
            r_planner_cum = cumsum_factor_matrix @ r_planner

            policy_loss = -torch.mean(train_batch[PreLearningProcessing.AWARENESS] * r_planner_cum)
            reward_cost = ((r_planner**2).sum(-1)**0.5).mean()

        model.tower_stats["planner_policy_loss"] = policy_loss
        model.tower_stats["planner_reward_cost"] = reward_cost

        total_loss = policy_loss + self.config['planner_reward_cost'] * reward_cost

        return total_loss

    def compute_central_planner_actions(self, sample_batch: SampleBatch) -> np.ndarray:
        """This function computes the r_planner, since we cannot truct on rllib's sample, because they might use exploration config.
        """
        if not self.is_central_planner:
            raise ValueError  # this can only happen to central planner

        copied_batch = self._lazy_tensor_dict(sample_batch.copy(shallow=False))  # this also convert batch to device
        dist_input, _ = self.model(copied_batch)
        action_dist = self.dist_class(dist_input, self.model)
        actions = action_dist.deterministic_sample()

        return actions.detach().cpu().numpy()

    def compute_awareness(self, sample_batch: SampleBatch) -> SampleBatch:
        """This funciton compute the 'awareness' of an agent.

            The awareness is a intermediate value used in total value function approximation. In the policy gradient approximation, 
                V_j = Prod_{i,t} pi_i(t) x R_j.
            and the gradient is 
                nabla_i V_j = (sum_t nabla_i log pi_i(t)) x R_j.
            and policy gradient theorem, we can replace R_j with advantage function
                nabla_i V_j = sum_t ( (nabla_i log pi_i(t)) x delta_{j, t} )
            Then the total value function's gradient is
                nabla_i V = sum_t ( (nabla_i log pi_i(t)) x (sum_j delta_{j, t}) )
            The AWARENESS is defined as
                a_{t,i} = (nabla_i V)^T x (nabla_i log pi_i(t))
                        = sum_t' ( (nabla_i log pi_i(t')) x (sum_j delta_{j, t'}) )^T x (nabla_i log pi_i(t))
            The awareness is used for latter update of central planner
                Delta theta_p = sum_{i, t} a_{t, i} x nabla_p R^p_{t, i}
            where R^p_{t,i} is the accumlated total reward.
        """
        # planner_reward_max means no planner's reward, so we skip this for faster computation
        if self.is_central_planner or self.config['planner_reward_max'] == 0.0:
            return sample_batch

        # get ready for calculating awareness
        copied_batch = self._lazy_tensor_dict(sample_batch.copy(shallow=False))  # this also convert batch to device
        awareness = torch.zeros(copied_batch[SampleBatch.REWARDS].shape, device=self.device)

        if self.config['param_assumption'] == 'neural':
            """
            Agents do policy gradient with assumtion of neural network parameterization
            """

            def func_call_log_probs_along_traj(params, input_batch: SampleBatch):
                dist_input, _ = functional_call(self.model, params, input_batch)
                action_dist = self.dist_class(dist_input, self.model)
                log_probs = action_dist.logp(input_batch[SampleBatch.ACTIONS]).reshape(-1)  # shape (Batch, )
                return log_probs

            dict_params = dict(self.model.named_parameters())
            jac_logp_theta = jacrev(
                lambda params: func_call_log_probs_along_traj(params=params, input_batch=copied_batch),
                argnums=0,
            )(dict_params)  # the jacobian is a dict with each element of shape (Batch, (shape_params))

            for param in dict_params.keys():
                """
                first accumlate over all advantage,
                chagne to ((shape_params), Batch) for broadcasting.
                NOTE 1: pairwise multiplication btw torch tensor and np array, tensor as front!
                NOTE 2: unlike a3c algorithm, I didnot consider the case of model is recurrent.
                """
                agent_pg = torch.sum(jac_logp_theta[param].movedim(0, -1) * copied_batch[PreLearningProcessing.TOTAL_ADVANTAGES].reshape(1, -1), dim=-1)  # (shape_params,)
                awareness_unsummed = jac_logp_theta[param] * agent_pg  # (Batch, (shape_params, ))
                awareness += awareness_unsummed.view(awareness_unsummed.shape[0], -1).sum(dim=-1)

        elif self.config['param_assumption'] == 'softmax':
            """
            Agents do policy gradient with assumtion of softmax parameterizization. In this case:
                p(a, s) = softmax(theta(a', s)) (a)
                paital{logp(a, s)}{theta(a', s')} = 1{s=s', a=a'} - pi(a', s)1{s=s'}
            Therefore, we only need to return jacobian(logp, theta) as tensor:
                1 - pi(a, s)
            of shape (Batch, num_actions).
            NOTE: 'softmax' option natually assumes a multinomial distribution, so the model's output is assumed to be the logits.
            NOTE: Determining state is equal or not is difficult for me, so I assume every input state is not equal, therefore, the result is simply g_log_p(s)^2.sum x total_advantage
            """
            logits, _ = self.model(copied_batch)  # shape is (Batch, num_actions)
            g_logp_s = 1 - torch.softmax(logits, dim=-1)  # (Batch, num_actions)
            awareness += (g_logp_s**2).sum(dim=-1) * copied_batch[PreLearningProcessing.TOTAL_ADVANTAGES].reshape(-1)

        elif self.config['param_assumption'] == 'softmax_single_state':
            """Agent with softmax parameterization assumption and all states are the same
            """
            logits, _ = self.model(copied_batch)  # shape is (Batch, num_actions)
            g_logp_s = 1 - torch.softmax(logits, dim=-1)  # (Batch, num_actions)

            pg = (g_logp_s * (copied_batch[PreLearningProcessing.TOTAL_ADVANTAGES].reshape(-1, 1))).sum(dim=0)
            awareness += (pg.reshape(1, -1) * g_logp_s).sum(dim=-1).reshape(-1)

        else:
            raise ValueError("The current policy parameterization assumption {} is not supported!!!".format(self.config['policy_param']))

        sample_batch[PreLearningProcessing.AWARENESS] = awareness.detach().cpu().numpy()

        return sample_batch
