import logging
import numpy as np
import time
from typing import Any, Callable, Dict, List, Optional, Type, Union, TYPE_CHECKING, Tuple
from ray.rllib.algorithms.a3c.a3c import A3CConfig
from ray.rllib.env.env_context import EnvContext
from ray.rllib.utils.from_config import NotProvided

from ray.util.debug import log_once
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.callbacks import DefaultCallbacks, MultiCallbacks
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided, Space, gym
from ray.rllib.algorithms.pg import PGConfig, PG
from ray.rllib.algorithms.a3c import A3CConfig, A3C
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
from ray.rllib.utils.typing import EnvConfigDict, MultiAgentPolicyConfigDict, ResultDict, SampleBatchType
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    SYNCH_WORKER_WEIGHTS_TIMER,
)
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.typing import (
    AgentID,
    AlgorithmConfigDict,
    ModelGradients,
    ModelWeights,
    PolicyID,
    PolicyState,
    T,
    EnvType,
    EnvCreator,
    TensorStructType,
    TensorType,
)

if TYPE_CHECKING:
    from ray.rllib.core.rl_module import RLModule

logger = logging.getLogger(__name__)

from gymnasium import spaces
from gymnasium.vector.utils import batch_space, concatenate, iterate, create_empty_array

from .constants import PreLearningProcessing, CENTRAL_PLANNER
from .callback import AMDDefualtCallback
from .wrappers import MultiAgentEnvWithCentralPlanner
from .utils import get_env_example, get_availability_mask


class AMDConfig(A3CConfig):
    """Defines a configuration class from which an Adaptive Mechanism Design algorithm can be built. It inhirits from A3C algorithm but not all setting is used, e.g. entropy or sample_async.
    
    Note that this algorithm use a callback, amd.callback.AMDDefaultCallback. If you provide some call backs, it will be excuted after this call back"""

    def __init__(self, algo_class=None):
        super().__init__(algo_class=algo_class or AMD)

        self.callbacks_class = AMDDefualtCallback
        self.policies = {}
        self.policy_mapping_fn = (lambda agent_id, episode, worker, **kwargs: agent_id)  # each agent id shares different policy

        # this is the default if .adaptive_mechanism_design
        self.agent_pseudo_lr = self.lr
        self.central_planner_lr = self.lr
        self.coop_agent_list = None
        self.param_assumption: str = 'neural'
        self.planner_reward_max: float = 1.0
        self.force_zero_sum: bool = False
        self.grad_on_reward: bool = False
        self.planner_reward_cost: float = 0.0

    @override(A3CConfig)
    def callbacks(self, callbacks_class) -> "AMDConfig":
        if callbacks_class is None:
            callbacks_class = DefaultCallbacks
        # Check, whether given `callbacks` is a callable.
        if not callable(callbacks_class):
            raise ValueError("`config.callbacks_class` must be a callable method that "
                             "returns a subclass of DefaultCallbacks, got "
                             f"{callbacks_class}!")
        self.callbacks_class = MultiCallbacks([AMDDefualtCallback, callbacks_class])

        return self

    @override(A3CConfig)
    def environment(self, **kwargs) -> AlgorithmConfig:
        """This also computes an env example and if coop_agent_list is not specified, turn is to all agents as coorperative agent"""
        super().environment(**kwargs)
        env_example = get_env_example(self)
        if (self.coop_agent_list is None) and isinstance(env_example, MultiAgentEnv):
            self.coop_agent_list = list(env_example.get_agent_ids())

        return self

    @override(A3CConfig)
    def training(
        self,
        *,
        agent_pseudo_lr: Optional[float] = NotProvided,
        central_planner_lr: Optional[float] = NotProvided,
        coop_agent_list: Optional[List[str]] = NotProvided,
        param_assumption: Optional[str] = NotProvided,
        planner_reward_max: Optional[float] = NotProvided,
        force_zero_sum: Optional[bool] = NotProvided,
        grad_on_reward: Optional[bool] = NotProvided,
        planner_reward_cost: Optional[float] = NotProvided,
        **kwargs,
    ) -> "AMDConfig":
        """
        agent_pseudo_lr: the learning rate of individual agents assumed by central planner, default to be equal to lr
        central_planner_lr: the learning rate of central planner, default to be lr TODO currently this is not supported
        coop_agent_list: a list of agent_id in the environment that is wished to be designed to be cooperative. For example, when agent includes prey and predator, we only wish preditors to be cooperative. This spetifies the agent in amd's loss.
        param_assumption: assump the policy to be parametrized by direct softmax or neural, this affect how the algoirthm calculate the awareness.
        planner_reward_max: [-R, R] for planner
        force_zero_sum: force the reward to minus its mean, in all cooperative agents.
        grad_on_reward, whether the reward can be pass gradient, TODO this is not used and for futher, set to be false
        planner_reward_cost: the loss penalty of using reward, TODO currently this is not supported since the loss cannot pass gradient from reward_planner
        """
        super().training(**kwargs)

        if agent_pseudo_lr is not NotProvided:
            self.agent_pseudo_lr = agent_pseudo_lr
        if central_planner_lr is not NotProvided:
            self.central_planner_lr = central_planner_lr
        if coop_agent_list is not NotProvided:
            self.coop_agent_list = coop_agent_list
        if param_assumption is not NotProvided:
            self.param_assumption = param_assumption
        if planner_reward_max is not NotProvided:
            self.planner_reward_max = planner_reward_max
        if force_zero_sum is not NotProvided:
            self.force_zero_sum = force_zero_sum
        if grad_on_reward is not NotProvided:
            self.grad_on_reward = grad_on_reward
        if planner_reward_cost is not NotProvided:
            self.planner_reward_cost = planner_reward_cost

        return self

    def get_multi_agent_setup(
        self,
        *,
        policies: MultiAgentPolicyConfigDict | None = None,
        env: MultiAgentEnv | None = None,
        spaces: Dict[PolicyID, Tuple[Space, Space]] | None = None,
        default_policy_class: Type[Policy] | None = None,
    ) -> Tuple[MultiAgentPolicyConfigDict, Callable[[PolicyID, SampleBatchType], bool]]:
        """Code copied from rllib, the basic logic is if no policies is provided, also"""

        # ! STEP 1: get the total class of policy
        # first if there is no central planner, turn the env into one with central planner
        # if env is none, we should return the multiagent version of env!!!
        env = (env or MultiAgentEnvWithCentralPlanner(get_env_example(self)))
        assert isinstance(env, MultiAgentEnv), "The current code only supports multiagent env!"
        env: MultiAgentEnv

        # if no given policy, use all agent ids
        policies = policies or env.get_agent_ids()

        policies, is_policy_to_train = super().get_multi_agent_setup(
            policies=policies,
            env=env,
            spaces=spaces,
            default_policy_class=default_policy_class,
        )

        return policies, is_policy_to_train


class AMD(A3C):
    """
        Training Iteration logic:
            1. workers sample {trajectories, r_real}
            2. forward {trajectories, r_real} to each agent, get V and g_log_p (forward_1)
            3. forward {trajectories} to planning agent, get r_planner, compute planner's loss -> do policy gradient update
            4. forward {r_planner, r_real} to each agent, compute loss -> do policy gradient update (forward_2)

        Therefore, we need two times of forwarding to agent. For forward_1, we can use a callback on_sample_end(), and and use keyword value_fn_estimate, reward_planner
    """

    reward_space_unflattened: spaces.Dict
    reward_space: spaces.Box

    @classmethod
    @override(A3C)
    def get_default_config(cls) -> AlgorithmConfig:
        return AMDConfig()

    @classmethod
    @override(A3C)
    def get_default_policy_class(cls, config: AlgorithmConfig) -> Type[Policy] | None:
        if config["framework"] == 'torch':
            from .amd_torch_policy import AMDAgentTorchPolicy
            return AMDAgentTorchPolicy
        else:
            raise NotImplementedError("Current algorithm only support PyTorch!")

    @override(A3C)
    def training_step(self) -> ResultDict:

        # NOTE: this part is copied from original code, and also used by policy gradient
        train_batch: SampleBatch
        if self.config.count_steps_by == "agent_steps":
            train_batch = synchronous_parallel_sample(worker_set=self.workers, max_agent_steps=self.config.train_batch_size)
        else:
            train_batch = synchronous_parallel_sample(worker_set=self.workers, max_env_steps=self.config.train_batch_size)
        train_batch = train_batch.as_multi_agent()
        self._counters[NUM_AGENT_STEPS_SAMPLED] += train_batch.agent_steps()
        self._counters[NUM_ENV_STEPS_SAMPLED] += train_batch.env_steps()
        # NOTE: copy end

        # NOTE: this is the start of my own code
        # for conviniency
        worker = self.workers.local_worker()
        policy_map = worker.policy_map

        # compute awareness
        if train_batch.agent_steps() > 0:
            for policy_id in self.config["coop_agent_list"]:
                if policy_id in policy_map.keys():
                    policy_map[policy_id].compute_awareness(train_batch[policy_id])

            train_batch = self.prelearning_process_trajectory(train_batch)
        # NOTE: this is the end of my own code

        # NOTE: this is the start of copied code
        train_results = {}
        if train_batch.agent_steps() > 0:
            # Use simple optimizer (only for multi-agent or tf-eager; all other
            # cases should use the multi-GPU optimizer, even if only using 1 GPU).
            # TODO: (sven) rename MultiGPUOptimizer into something more
            #  meaningful.
            if self.config._enable_rl_trainer_api:
                train_results = self.trainer_runner.update(train_batch)
            elif self.config.get("simple_optimizer") is True:
                train_results = train_one_step(self, train_batch)
            else:
                train_results = multi_gpu_train_one_step(self, train_batch)
        else:
            # Wait 1 sec before probing again via weight syncing.
            time.sleep(1)

        # Update weights and global_vars - after learning on the local worker - on all
        # remote workers (only those policies that were actually trained).
        global_vars = {
            "timestep": self._counters[NUM_ENV_STEPS_SAMPLED],
        }
        with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
            # TODO (Avnish): Implement this on trainer_runner.get_weights().
            # TODO (Kourosh): figure out how we are going to sync MARLModule
            # weights to MARLModule weights under the policy_map objects?
            from_worker = None
            if self.config._enable_rl_trainer_api:
                from_worker = self.trainer_runner
            self.workers.sync_weights(
                from_worker=from_worker,
                policies=list(train_results.keys()),
                global_vars=global_vars,
            )
        # NOTE: copy end

        return train_results

    def prelearning_process_trajectory(self, train_batch: SampleBatch):
        """This fucntion organizes train batch"""
        # ! STEP 1: get central planer's t, episode_id
        cp_t = train_batch[CENTRAL_PLANNER][SampleBatch.T]
        cp_eps_id = train_batch[CENTRAL_PLANNER][SampleBatch.EPS_ID]

        # ! STEP 2: get a mask of array indicating which agent are in compuation
        appearance_mask_unflattened = {}  # stores which agent shows up in dict, to minus its mean
        for agent_id in self.reward_space_unflattened.keys():
            appearance_mask_unflattened[agent_id] = (agent_id in self.config['coop_agent_list'] or (self.config['coop_agent_list'] is None))
        appearance: np.ndarray = spaces.flatten(self.reward_space_unflattened, appearance_mask_unflattened).astype(bool)  # shape of (n_agent, )

        # ! STEP 3: preprocess and distribute rewards
        r_planner: np.ndarray = self.config['planner_reward_max'] * train_batch[CENTRAL_PLANNER][SampleBatch.ACTIONS]
        if self.config['force_zero_sum']:
            r_planner = r_planner - r_planner[:, appearance].mean(axis=-1).reshape(-1, 1)
        train_batch[CENTRAL_PLANNER][PreLearningProcessing.R_PLANNER] = r_planner

        batch_size = r_planner.shape[0]
        batch_uf = batch_space(self.reward_space_unflattened, batch_size)
        r_planner_unflattened = spaces.unflatten(batch_uf, r_planner.T.flatten())  # this transpose is very important!

        # pre-define structured data for central planner
        awareness_uf = create_empty_array(self.reward_space_unflattened, batch_size)
        avail_uf = create_empty_array(self.reward_space_unflattened, batch_size)

        # ! STEP 4: calculate availability, put r_planner into agent, put avail and awareness into spaces.Dict
        policy_map = self.workers.local_worker().policy_map
        for policy_id in self.config["coop_agent_list"]:
            if policy_id in policy_map.keys():
                batch = train_batch[policy_id]

                # 4.1 compute availability
                avail = get_availability_mask(
                    cp_t=cp_t,
                    cp_eps_id=cp_eps_id,
                    ag_t=batch[SampleBatch.T],
                    ag_eps_id=batch[SampleBatch.EPS_ID],
                )

                # 4.2 put r_planner into agent's batch
                batch[PreLearningProcessing.R_PLANNER] = r_planner_unflattened[policy_id].reshape(-1)[avail]

                # modify availability
                avail_uf[policy_id] = avail.reshape(-1, 1).astype(float)

                # modify awareness
                awareness_uf[policy_id][avail, :] = batch[PreLearningProcessing.AWARENESS].reshape(-1, 1)

        # ! STEP 5: change awareness and avail_f into compact shape
        awareness = spaces.flatten(batch_uf, awareness_uf).reshape(r_planner.T.shape).T  # the transpose is necessary
        train_batch[CENTRAL_PLANNER][PreLearningProcessing.AWARENESS] = awareness
        availability = spaces.flatten(batch_uf, avail_uf).reshape(r_planner.T.shape).T.astype(bool)
        train_batch[CENTRAL_PLANNER][PreLearningProcessing.AVAILABILITY] = availability

        return train_batch

    @override(A3C)
    @staticmethod
    def _get_env_id_and_creator(env_specifier: Union[str, EnvType, None], config: AlgorithmConfig) -> Tuple[Optional[str], EnvCreator]:
        """This function creats a wrapped environment with central planner"""
        env_id, env_creator = Algorithm._get_env_id_and_creator(env_specifier=env_specifier, config=config)

        return env_id, lambda cfg: MultiAgentEnvWithCentralPlanner(env=env_creator(cfg))
