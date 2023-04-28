"""
PyTorch policy class used for AMD.
"""
import logging
from typing import Dict, List, Type, Union, Optional, Tuple, Any

from ray.rllib.evaluation.episode import Episode
from ray.rllib.utils.typing import AgentID
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.algorithms.pg.pg import PGConfig
from ray.rllib.algorithms.pg.utils import post_process_advantages
from ray.rllib.evaluation.postprocessing import Postprocessing, compute_gae_for_sample_batch
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_mixins import LearningRateSchedule, ValueNetworkMixin
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType
from ray.rllib.utils.torch_utils import apply_grad_clipping, sequence_mask
from ray.rllib.algorithms.a3c.a3c_torch_policy import A3CTorchPolicy
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2

from .constants import PreLearningProcessing, CENTRAL_PLANNER

import torch
import torch.nn as nn
from torch.func import jacrev, functional_call

logger = logging.getLogger(__name__)

from gymnasium import spaces
from gymnasium.vector.utils import concatenate, create_empty_array
import numpy as np


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

    def __init__(self, observation_space, action_space, config):
        # TODO remove this: put it into config

        super().__init__(observation_space, action_space, config)

        # TODO remove this: put it into config
        self.config['policy_param'] = 'neural'

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
        sample_batch = super().postprocess_trajectory(sample_batch)

        if not self.is_central_planner:
            # ! STEP 1: calculate gae, this will add ADVANTAGEs and VALUE_TAEGETS into this batch
            # central planner does not need critic
            sample_batch = compute_gae_for_sample_batch(self, sample_batch, other_agent_batches, episode)

            # ! STEP 2: placeholder for reward from planner, this is for skipping Policy._initialize_loss_from_dummy_batch().
            # Since the default class is agent, we only have to pass a r_planner term
            sample_batch[PreLearningProcessing.R_PLANNER] = 0 * sample_batch[SampleBatch.REWARDS]

        return sample_batch

    def loss_agent(
        self,
        model: ModelV2,
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

        # total rewards = GAE of critic + reward by planner
        total_rewards = train_batch[Postprocessing.ADVANTAGES] + train_batch[PreLearningProcessing.R_PLANNER]

        # Final policy loss.
        policy_loss = -torch.mean(log_probs * total_rewards)

        # Compute a value function loss.
        if self.config["use_critic"]:
            value_loss = 0.5 * torch.sum(torch.pow(values.reshape(-1) - train_batch[Postprocessing.VALUE_TARGETS], 2.0))

        else:  # Ignore the value function.
            value_loss = 0.0

        # Store values for stats function in model (tower), such that for
        # multi-GPU, we do not override them during the parallel loss phase.
        model.tower_stats["agent_policy_loss"] = policy_loss
        model.tower_stats["agent_value_loss"] = value_loss

        return policy_loss

    def preprocess_batch_before_learning(
        self,
        sample_batch: SampleBatch,
        other_agent_batches: Dict[Any, SampleBatch] | None = None,
        episode: Episode | None = None,
    ) -> SampleBatch:
        sample_batch = super().postprocess_trajectory(sample_batch)

        raise NotImplementedError

        self.config['policy_param'] = 'neural'  # ! For testing

        # ! STEP 1: get awareness, depending on algorithm's parameterizaiton assumption, whether it is neural param, or softmax parameterization
        # get ready for calculating awareness
        copied_batch = self._lazy_tensor_dict(sample_batch.copy(shallow=False))  # this also convert batch to device
        awareness = torch.zeros(copied_batch[SampleBatch.REWARDS].shape, device=self.device)
        if self.config['policy_param'] == 'neural':
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
            )(dict_params)  # a dict with each element of shape (Batch, (shape_params))

            for param in dict_params.keys():
                """
                first accumlate over all advantage,
                chagne to ((shape_params), Batch) for broadcasting.
                NOTE 1: multiplication btw torch tensor and np array, tensor as front!
                NOTE 2: unlike a3c algorithm, I didnot consider the case of model is recurrent.
                """
                agent_pg = torch.sum(jac_logp_theta[param].movedim(0, -1) * copied_batch[Postprocessing.ADVANTAGES], dim=-1)  # (shape_params,)
                awareness_unsummed = agent_pg * jac_logp_theta[param]  # (Batch, (shape_params))
                awareness = awareness + awareness_unsummed.view(awareness_unsummed.shape[0], -1).sum(dim=-1)

        elif self.config['policy_param'] == 'softmax':
            """
            Agents do policy gradient with assumtion of softmax parameterizization. In this case:
                paital{logp(a, s)}{theta(a', s')} = 1{s=s', a=a'} - pi(a', s)1{s=s'}
            Therefore, we only need to return jacobian(logp, theta) as tensor:
                1 - pi(a, s)
            of shape (Batch, num_actions).
            NOTE: 'softmax' option natually assumes a multinomial distribution, so the model's output is assumed to be the logits.
            NOTE: Determining state is equal or not is difficult for me, so I assume every input state is not equal, therefore, the result is simply g_log_p(s)^2.sum x advantage
            """
            logits, _ = self.model(copied_batch)  # shape is (Batch, num_actions)
            g_logp_s = 1 - torch.softmax(logits, dim=-1)  # (Batch, num_actions)
            awareness = (g_logp_s * g_logp_s).sum(dim=1) * copied_batch[Postprocessing.ADVANTAGES]
        else:
            raise ValueError("The current policy parameterization assumption {} is not supported!!!".format(self.config['policy_param']))

        sample_batch[PreLearningProcessing.AWARENESS] = awareness.detach().cpu().numpy()

        return sample_batch
