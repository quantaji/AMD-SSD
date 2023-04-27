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
from .amd import PreLearningProcessing

from .amd import AMDAgentPolicy

import torch
import torch.nn as nn
from torch.func import jacrev, functional_call

logger = logging.getLogger(__name__)


class AMDAgentTorchPolicy(A3CTorchPolicy, AMDAgentPolicy):
    """Pytorch policy class used for Adaptive Mechanism design."""

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        self.config['policy_param'] = 'neural'

    model: nn.Module

    def postprocess_trajectory(
        self,
        sample_batch: SampleBatch,
        other_agent_batches: Dict[Any, SampleBatch] | None = None,
        episode: Episode | None = None,
    ) -> SampleBatch:
        sample_batch = super().postprocess_trajectory(sample_batch)

        # ! STEP 1: calculate gae, this will add ADVANTAGEs and VALUE_TAEGETS into this batch
        sample_batch = compute_gae_for_sample_batch(self, sample_batch, other_agent_batches, episode)

        self.config['policy_param'] = 'neural'  # ! For testing

        # ! STEP 2: get awareness, depending on algorithm's parameterizaiton assumption, whether it is neural param, or softmax parameterization
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


class AMDPlannerTorchPolicy(LearningRateSchedule, TorchPolicyV2):
    """Pytorch policy class used for Adaptive Mechanism design."""

    pass
