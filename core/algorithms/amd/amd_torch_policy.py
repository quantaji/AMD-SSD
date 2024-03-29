"""
PyTorch policy class used for AMD.
"""
from typing import Any, Dict, List, Union

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
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.policy import Policy
from ray.rllib.policy.rnn_sequencing import pad_batch_to_sequences_of_same_size
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.torch_utils import sequence_mask
from ray.rllib.utils.typing import TensorType
from torch.func import functional_call, jacrev

from .constants import PreLearningProcessing
from .utils import action_to_reward, cumsum_factor_across_eps


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

    def compute_central_planner_actions(self, sample_batch: SampleBatch) -> np.ndarray:
        """This function computes the r_planner, since we cannot truct on rllib's sample, because they might use exploration config.
        """
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
                "policy_loss": torch.mean(torch.stack(self.get_tower_stats("policy_loss"))),
                "vf_loss": torch.mean(torch.stack(self.get_tower_stats("vf_loss"))),
                "policy_entropy": torch.mean(torch.stack(self.get_tower_stats("policy_entropy"))),
                "cur_lr": self.cur_lr,
                "entropy_coeff": self.entropy_coeff,
                "policy_loss": torch.mean(torch.stack(self.get_tower_stats("policy_loss"))),
                "amd_loss": torch.mean(torch.stack(self.get_tower_stats("amd_loss"))),
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
            sample_batch[PreLearningProcessing.AVAILABILITY] = 0 * sample_batch[SampleBatch.REWARDS]
        else:
            sample_batch[SampleBatch.OBS] = sample_batch[SampleBatch.NEXT_OBS]  # shift back

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

        if self.is_recurrent():
            B = len(train_batch[SampleBatch.SEQ_LENS])
            max_seq_len = dist_inputs.shape[0] // B
            mask_orig = sequence_mask(train_batch[SampleBatch.SEQ_LENS], max_seq_len)
            valid_mask = torch.reshape(mask_orig, [-1])
        else:
            valid_mask = torch.ones_like(values, dtype=torch.bool)

        # Create an action distribution object.
        action_dist = dist_class(dist_inputs[valid_mask], model)

        # Calculate the vanilla PG loss based on: L = -E[ log(pi(a|s)) * A]
        log_probs = action_dist.logp(train_batch[SampleBatch.ACTIONS][valid_mask])

        # the loss contains two parts, first the real rewards, by critit
        # and the second: the reward from central planner
        # # ps: if this agent is not in cooperative list, its r_planner_cum is zero

        # Final policy loss.
        policy_loss = -torch.mean(log_probs * train_batch[Postprocessing.ADVANTAGES][valid_mask])
        amd_loss = -torch.mean(log_probs * train_batch[PreLearningProcessing.R_PLANNER_CUM][valid_mask])

        # Compute a value function loss.
        if self.config["use_critic"]:
            value_loss = 0.5 * torch.mean(torch.pow(
                values.reshape(-1)[valid_mask] - train_batch[Postprocessing.VALUE_TARGETS][valid_mask],
                2.0,
            ))
        else:  # Ignore the value function.
            value_loss = 0.0

        entropy = torch.sum(action_dist.entropy())

        # Store values for stats function in model (tower), such that for
        # multi-GPU, we do not override them during the parallel loss phase.
        model.tower_stats["policy_loss"] = policy_loss
        model.tower_stats["vf_loss"] = value_loss
        model.tower_stats["policy_entropy"] = entropy
        model.tower_stats["amd_loss"] = amd_loss

        total_loss = policy_loss + amd_loss + self.config['vf_loss_coeff'] * value_loss - entropy * self.entropy_coeff

        return total_loss

    def loss_central_planner(
        self,
        model: torch_modelv2,
        dist_class: ActionDistribution,
        train_batch: SampleBatch,
    ) -> TensorType | List[TensorType]:

        # Pass the training data through our model to get distribution parameters.
        dist_inputs, _ = model(train_batch)

        if self.is_recurrent():
            B = len(train_batch[SampleBatch.SEQ_LENS])
            max_seq_len = train_batch[SampleBatch.REWARDS].shape[0] // B
            mask_orig = sequence_mask(train_batch[SampleBatch.SEQ_LENS], max_seq_len)
            valid_mask = torch.reshape(mask_orig, [-1])
        else:
            valid_mask = torch.ones_like(train_batch[SampleBatch.REWARDS], dtype=torch.bool)

        policy_loss = torch.zeros_like(dist_inputs, requires_grad=True).mean()
        reward_cost = torch.zeros_like(policy_loss, requires_grad=True)

        # only compute loss when reward_max is not zero, for saving computation
        if self.config['planner_reward_max'] > 0.0:

            # Create an action distribution object.
            action_dist = dist_class(dist_inputs[valid_mask], model)

            # Compute the actual actions
            actions = action_dist.deterministic_sample()

            # get reward from actions
            r_planner = action_to_reward(
                actions=actions,
                appearance=self.appearance,
                reward_max=self.config['planner_reward_max'],
                zero_sum=self.config['force_zero_sum'],
                availability=train_batch[PreLearningProcessing.AVAILABILITY][valid_mask],
            )

            # get the matrix for computing accumulated reward
            # fix: somehow on clusters, discoutned factor matrix is not T x T
            cp_eps_id = train_batch[SampleBatch.EPS_ID][valid_mask]

            # we project it onto per episode scale
            factor_matrix = cumsum_factor_across_eps(cp_eps_id).to(r_planner)
            r_planner_cum = factor_matrix @ r_planner
            awareness_cum = factor_matrix @ train_batch[PreLearningProcessing.AWARENESS][valid_mask]

            policy_loss = -torch.mean(awareness_cum * r_planner_cum) * self.config['agent_pseudo_lr']
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

        # I found there is some stange post processing for rnn...
        copied_batch = sample_batch.copy(shallow=False)
        if not copied_batch.zero_padded:
            pad_batch_to_sequences_of_same_size(
                batch=copied_batch,
                max_seq_len=self.max_seq_len,
                shuffle=False,
                batch_divisibility_req=self.batch_divisibility_req,
                view_requirements=self.view_requirements,
            )
        copied_batch = self._lazy_tensor_dict(copied_batch)  # this also convert batch to device

        if self.is_recurrent():
            B = len(copied_batch[SampleBatch.SEQ_LENS])
            max_seq_len = copied_batch[SampleBatch.REWARDS].shape[0] // B
            mask_orig = sequence_mask(copied_batch[SampleBatch.SEQ_LENS], max_seq_len)
            valid_mask = torch.reshape(mask_orig, [-1])
        else:
            valid_mask = torch.ones_like(copied_batch[SampleBatch.REWARDS], dtype=torch.bool)

        dist_inputs, _ = self.model(copied_batch)
        action_dist = self.dist_class(dist_inputs[valid_mask], self.model)
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
        copied_batch = sample_batch.copy(shallow=False)
        if not copied_batch.zero_padded:
            pad_batch_to_sequences_of_same_size(
                batch=copied_batch,
                max_seq_len=self.max_seq_len,
                shuffle=False,
                batch_divisibility_req=self.batch_divisibility_req,
                view_requirements=self.view_requirements,
            )
        copied_batch = self._lazy_tensor_dict(copied_batch)  # this also convert batch to device

        if self.is_recurrent():
            B = len(copied_batch[SampleBatch.SEQ_LENS])
            max_seq_len = copied_batch[SampleBatch.REWARDS].shape[0] // B
            mask_orig = sequence_mask(copied_batch[SampleBatch.SEQ_LENS], max_seq_len)
            valid_mask = torch.reshape(mask_orig, [-1])
        else:
            valid_mask = torch.ones_like(copied_batch[SampleBatch.REWARDS], dtype=torch.bool)

        awareness = torch.zeros(copied_batch[SampleBatch.REWARDS][valid_mask].shape, device=self.device)

        if self.config['param_assumption'] == 'neural':
            """
            Agents do policy gradient with assumtion of neural network parameterization
            """

            def func_call_log_probs_along_traj(params, input_batch: SampleBatch):
                dist_inputs, _ = functional_call(self.model, params, input_batch)
                action_dist = self.dist_class(dist_inputs, self.model)
                log_probs = action_dist.logp(input_batch[SampleBatch.ACTIONS]).reshape(-1)  # shape (Batch, )
                return log_probs[valid_mask]

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
                agent_pg = torch.sum(jac_logp_theta[param].movedim(0, -1) * copied_batch[PreLearningProcessing.TOTAL_ADVANTAGES][valid_mask].reshape(1, -1), dim=-1)  # (shape_params,)
                awareness_unsummed = jac_logp_theta[param] * agent_pg  # (Batch, (shape_params, ))
                awareness += awareness_unsummed.view(awareness_unsummed.shape[0], -1).sum(dim=-1)

        elif self.config['param_assumption'] == 'softmax':
            """
            Agents do policy gradient with assumtion of softmax parameterizization. In this case:
                p(a, s) = softmax(theta(a', s)) (a)
                paital{logp(a, s)}{theta(a', s')} = 1{s=s', a=a'} - pi(a', s)1{s=s'}
            NOTE: 'softmax' option natually assumes a multinomial distribution, so the model's output is assumed to be the logits.
            NOTE: Determining state is equal or not is difficult. Therefore, I use hash function over observation.
            Derivation:
                awareness(tau) := (sum_t nabla_theta logpi(a_t, s_t) * delta_t) ^T nabla_theta logpi(a_tau, s_tau)
                = sum_{t, a', s'} delta_t x 1{s_t = s'} x (1{a_t = a'} - pi(a', s_t)) x 1{s_tau = s'} x (1{a_tau = a'} - pi(a', s_tau))
            We define j_t = 1{a_t = a'} - pi(a', s_t) in R^{|A|} as a vector of dimension of actiona space, then
                awareness(tau) = sum_{t, s'} delta_t x 1{s_t = s'} x 1{s_tau = s'} x (j_t ^T j_tau)
                =  sum_{t} delta_t x 1{s_t = s_tau} x (j_t ^T j_tau)
                = M x delta
            M_{i,j} := 1{s_i = s_j} x (j_i ^T j_j) as a TxT matrix, and delta = [delta_t] is a Tx1 matrix
            """
            assert self.dist_class == TorchCategorical, "Softmax parameterization assumtion assumse tabular case, where action is discrete and therefore, action distribution class is catagorical!"

            with torch.no_grad():
                logits, _ = self.model(copied_batch)  # shape is (Batch, num_actions)

                num_actions = logits.shape[1]
                jac_logp_s = torch.nn.functional.one_hot(copied_batch[SampleBatch.ACTIONS][valid_mask].long(), num_classes=num_actions) - torch.softmax(logits[valid_mask], dim=-1)  # (Batch, num_actions)
                jac_logp_s = jac_logp_s.detach().cpu().numpy()

                obs = copied_batch[SampleBatch.OBS][valid_mask].detach().cpu().numpy()
                obs_hash = np.apply_along_axis(
                    lambda arr: hash(arr.data.tobytes()),
                    1,
                    obs.reshape(obs.shape[0], -1),
                )  # this funciton computes the observation's hash

                # M = (jac_logp_s @ jac_logp_s.T) * torch.from_numpy((obs_hash.reshape(1, -1) == obs_hash.reshape(-1, 1))).to(jac_logp_s)
                # big matrix, we should work on cpu
                M = (jac_logp_s @ jac_logp_s.T) * (obs_hash.reshape(1, -1) == obs_hash.reshape(-1, 1))
                total_adv = copied_batch[PreLearningProcessing.TOTAL_ADVANTAGES][valid_mask].reshape(-1, 1).detach().cpu().numpy()

            awareness = awareness + torch.from_numpy((M @ total_adv).reshape(-1)).to(awareness)

        else:
            raise ValueError("The current policy parameterization assumption {} is not supported!!!".format(self.config['policy_param']))

        sample_batch[PreLearningProcessing.AWARENESS] = awareness.detach().cpu().numpy()

        return sample_batch
