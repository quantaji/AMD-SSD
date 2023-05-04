"""
PyTorch policy class used for AMD.
"""
import logging
from typing import Dict, List, Type, Union, Optional, Tuple

from ray.rllib.evaluation.episode import Episode
from ray.rllib.utils.typing import AgentID
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.algorithms.pg.pg import PGConfig
from ray.rllib.algorithms.pg.utils import post_process_advantages
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_mixins import LearningRateSchedule
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType

torch, nn = try_import_torch()

logger = logging.getLogger(__name__)


class AMDTorchPolicy(LearningRateSchedule, TorchPolicyV2):
    """Pytorch policy class used for Adaptive Mechanism design."""

    pass
