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
from ray.rllib.policy.sample_batch import SampleBatch



if TYPE_CHECKING:
    from ray.rllib.core.rl_module import RLModule

logger = logging.getLogger(__name__)


class PreLearningProcessing:
    """Constant definition for Sample Batch keywords before feeding to policy to learn"""

    AWARENESS = "agent_awareness"
    R_PLANNER = "reward_by_planner"
    AVAILABILITY = 'availability'  # used for central planner to get only the agent's reward when it is not terminated


class AMDAgentPolicy(Policy):

    def compute_gae_and_awareness(self, sample_batch: SampleBatch):
        """Calculate advantage function from critic, and also calculate awareness w.r.t. the whole batch"""
        raise NotImplementedError


class AMDConfig(PGConfig):
    pass


class AMD(PG):
    """
        Training Iteration logic:
            1. workers sample {trajectories, r_real}
            2. forward {trajectories, r_real} to each agent, get V and g_log_p (forward_1)
            3. forward {trajectories} to planning agent, get r_planner, compute planner's loss -> do policy gradient update
            4. forward {r_planner, r_real} to each agent, compute loss -> do policy gradient update (forward_2)

        Therefore, we need two times of forwarding to agent. For forward_1, we can use a callback on_sample_end(), and and use keyword value_fn_estimate, reward_planner
    """

    @classmethod
    @override(Algorithm)
    def get_default_policy_class(cls, config: AlgorithmConfig) -> Type[Policy] | None:
        if config["framework"] == 'torch':
            from .amd_torch_policy import AMDAgentTorchPolicy
            return AMDAgentTorchPolicy
        else:
            raise NotImplementedError("Current algorithm only support PyTorch!")
