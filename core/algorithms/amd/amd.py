import logging
from typing import List, Optional, Type, Union, TYPE_CHECKING

from ray.util.debug import log_once
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.algorithms.pg import PGConfig, PG
from ray.rllib.execution.rollout_ops import (
    standardize_fields, )
from ray.rllib.execution.train_ops import (
    train_one_step,
    multi_gpu_train_one_step,
)
from ray.rllib.utils.annotations import ExperimentalAPI
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import (
    Deprecated,
    DEPRECATED_VALUE,
    deprecation_warning,
)
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.typing import ResultDict
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    SYNCH_WORKER_WEIGHTS_TIMER,
)

if TYPE_CHECKING:
    from ray.rllib.core.rl_module import RLModule

logger = logging.getLogger(__name__)


class AMDConfig(PGConfig):
    pass


class AMD(Algorithm):

    @classmethod
    @override(Algorithm)
    def get_default_policy_class(cls, config: AlgorithmConfig) -> Type[Policy] | None:
        if config["framework"] == 'torch':
            from .amd_torch_policy import AMDTorchPolicy

            return AMDTorchPolicy
        else:
            raise NotImplementedError("Current algorithm only support PyTorch!")
