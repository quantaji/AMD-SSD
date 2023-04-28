from collections import defaultdict
import concurrent
import copy
from datetime import datetime
import functools
import gymnasium as gym
import importlib
import json
import logging
import numpy as np
import os
from packaging import version
import pkg_resources
import re
import tempfile
import time
import tree  # pip install dm_tree
from typing import (
    Callable,
    Container,
    DefaultDict,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

import ray
from ray._private.usage.usage_lib import TagKey, record_extra_usage_tag
from ray.actor import ActorHandle
from ray.air.checkpoint import Checkpoint
import ray.cloudpickle as pickle

from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.core.rl_module.marl_module import (
    MultiAgentRLModuleSpec,
    MultiAgentRLModule,
)

from ray.rllib.connectors.agent.obs_preproc import ObsPreprocessorConnector
from ray.rllib.algorithms.registry import ALGORITHMS as ALL_ALGORITHMS
from ray.rllib.env.env_context import EnvContext
from ray.rllib.env.utils import _gym_env_creator
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.metrics import (
    collect_episodes,
    collect_metrics,
    summarize_episodes,
)
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.execution.common import (
    STEPS_TRAINED_THIS_ITER_COUNTER,  # TODO: Backward compatibility.
)
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.execution.train_ops import multi_gpu_train_one_step, train_one_step
from ray.rllib.offline import get_dataset_and_shards
from ray.rllib.offline.estimators import (
    OffPolicyEstimator,
    ImportanceSampling,
    WeightedImportanceSampling,
    DirectMethod,
    DoublyRobust,
)
from ray.rllib.offline.offline_evaluation_utils import remove_time_dim
from ray.rllib.offline.offline_evaluator import OfflineEvaluator
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID, SampleBatch, concat_samples
from ray.rllib.utils import deep_update, FilterManager
from ray.rllib.utils.annotations import (
    DeveloperAPI,
    ExperimentalAPI,
    OverrideToImplementCustomLogic,
    OverrideToImplementCustomLogic_CallToSuperRecommended,
    PublicAPI,
    override,
)
from ray.rllib.utils.checkpoints import CHECKPOINT_VERSION, get_checkpoint_info
from ray.rllib.utils.debug import update_global_seed_if_necessary
from ray.rllib.utils.deprecation import (
    DEPRECATED_VALUE,
    Deprecated,
    deprecation_warning,
)
from ray.rllib.utils.error import ERR_MSG_INVALID_ENV_DESCRIPTOR, EnvError
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED,
    NUM_AGENT_STEPS_SAMPLED_THIS_ITER,
    NUM_AGENT_STEPS_TRAINED,
    NUM_ENV_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED_THIS_ITER,
    NUM_ENV_STEPS_TRAINED,
    SYNCH_WORKER_WEIGHTS_TIMER,
    TRAINING_ITERATION_TIMER,
)
from ray.rllib.utils.metrics.learner_info import LEARNER_INFO
from ray.rllib.utils.policy import validate_policy_id
from ray.rllib.utils.replay_buffers import MultiAgentReplayBuffer, ReplayBuffer
from ray.rllib.utils.spaces import space_utils
from ray.rllib.utils.typing import (
    AgentConnectorDataType,
    AgentID,
    AlgorithmConfigDict,
    EnvCreator,
    EnvInfoDict,
    EnvType,
    EpisodeID,
    PartialAlgorithmConfigDict,
    PolicyID,
    PolicyState,
    ResultDict,
    SampleBatchType,
    TensorStructType,
    TensorType,
)
from ray.tune.execution.placement_groups import PlacementGroupFactory
from ray.tune.experiment.trial import ExportFormat
from ray.tune.logger import Logger, UnifiedLogger
from ray.tune.registry import ENV_CREATOR, _global_registry
from ray.tune.resources import Resources
from ray.tune.result import DEFAULT_RESULTS_DIR
from ray.tune.trainable import Trainable
from ray.util import log_once
from ray.util.timer import _Timer
from ray.tune.registry import get_trainable_cls

from ray.rllib.env.env_context import EnvContext

tf1, tf, tfv = try_import_tf()

logger = logging.getLogger(__name__)


def get_env_example(config: AlgorithmConfig) -> EnvType:
    """These code are copied and modified from Algoirhtm._get_env_id_and_creator"""
    env_specifier = config.env
    env_creator: EnvCreator = None
    if isinstance(env_specifier, str):
        # An already registered env.
        if _global_registry.contains(ENV_CREATOR, env_specifier):
            env_creator = _global_registry.get(ENV_CREATOR, env_specifier)

        # A class path specifier.
        elif "." in env_specifier:

            def env_creator_from_classpath(env_context):
                try:
                    env_obj = from_config(env_specifier, env_context)
                except ValueError:
                    raise EnvError(ERR_MSG_INVALID_ENV_DESCRIPTOR.format(env_specifier))
                return env_obj

            env_creator = env_creator_from_classpath
        # Try gym/PyBullet/Vizdoom.
        else:
            env_creator = functools.partial(_gym_env_creator, env_descriptor=env_specifier)

    elif isinstance(env_specifier, type):
        env_id = env_specifier  # .__name__

        if config["remote_worker_envs"]:
            # Check gym version (0.22 or higher?).
            # If > 0.21, can't perform auto-wrapping of the given class as this
            # would lead to a pickle error.
            gym_version = pkg_resources.get_distribution("gym").version
            if version.parse(gym_version) >= version.parse("0.22"):
                raise ValueError("Cannot specify a gym.Env class via `config.env` while setting "
                                 "`config.remote_worker_env=True` AND your gym version is >= "
                                 "0.22! Try installing an older version of gym or set `config."
                                 "remote_worker_env=False`.")

            @ray.remote(num_cpus=1)
            class _wrapper(env_specifier):
                # Add convenience `_get_spaces` and `_is_multi_agent`
                # methods:
                def _get_spaces(self):
                    return self.observation_space, self.action_space

                def _is_multi_agent(self):
                    from ray.rllib.env.multi_agent_env import MultiAgentEnv

                    return isinstance(self, MultiAgentEnv)

            env_creator = lambda cfg: _wrapper.remote(cfg)
        # gym.Env-subclass: Also go through our RLlib gym-creator.
        elif issubclass(env_specifier, gym.Env):
            env_creator = functools.partial(
                _gym_env_creator,
                env_descriptor=env_specifier,
                auto_wrap_old_gym_envs=config.get("auto_wrap_old_gym_envs", True),
            )
        # All other env classes: Call c'tor directly.
        else:
            env_creator = lambda cfg: env_specifier(cfg)

    env_context = EnvContext(config.env_config, worker_index=0)

    env = env_creator(env_context)

    return env


def get_availability_mask(cp_t: np.ndarray, cp_eps_id: np.ndarray, ag_t: np.ndarray, ag_eps_id: np.ndarray):
    """Some times some agent dies but planner does not. So We have to compute a maske of what time step each agent is available. This function assumes that when agent dies, its batch does not have the correspongding time step t.
    """
    dtype = {'names': ['t', 'eps_id'], 'formats': [int, int]}

    cp_comb = np.vstack([cp_eps_id, cp_t]).T.copy().view(dtype).reshape(-1)
    ag_comb = np.vstack([ag_eps_id, ag_t]).T.copy().view(dtype).reshape(-1)

    # central planner have more time instance than agent so we don't need to union
    mask = np.in1d(cp_comb, ag_comb, assume_unique=True)

    return mask
