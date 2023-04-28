import logging
from typing import Callable, Dict, List, Optional, Type, Union, TYPE_CHECKING, Tuple
from ray.rllib.algorithms.a3c.a3c import A3CConfig

from ray.util.debug import log_once
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.callbacks import DefaultCallbacks, MultiCallbacks
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided, Space
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
from ray.rllib.utils.typing import MultiAgentPolicyConfigDict, ResultDict, SampleBatchType
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    SYNCH_WORKER_WEIGHTS_TIMER,
)
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.models.action_dist import ActionDistribution
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

from .constants import PreLearningProcessing, CENTRAL_PLANNER
from .callback import AMDDefualtCallback
from .wrappers import MultiAgentEnvWithCentralPlanner


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
        assert isinstance(env, MultiAgentEnv), "The current code only supports multiagent env!"
        env: MultiAgentEnv
        # we also assume that the env already has a central planner
        # if no given policy, use all agent ids
        policies = policies or env.get_agent_ids()

        policies, is_policy_to_train = super().get_multi_agent_setup(
            policies=policies,
            env=env,
            spaces=spaces,
            default_policy_class=default_policy_class,
        )

        # print(policies, is_policy_to_train)
        # for policy_id in policies.keys():
        #     # print(policy_id, policies[policy_id].observation_space, policies[policy_id].action_space)
        #     # policies[policy_id].policy_id

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

        # this part is copied from original code, and also used by policy gradient
        train_batch: SampleBatch
        if self.config.count_steps_by == "agent_steps":
            train_batch = synchronous_parallel_sample(worker_set=self.workers, max_agent_steps=self.config.train_batch_size)
        else:
            train_batch = synchronous_parallel_sample(worker_set=self.workers, max_env_steps=self.config.train_batch_size)
        train_batch = train_batch.as_multi_agent()
        self._counters[NUM_AGENT_STEPS_SAMPLED] += train_batch.agent_steps()
        self._counters[NUM_ENV_STEPS_SAMPLED] += train_batch.env_steps()

        # for pid, batch in train_batch.policy_batches.items():
        #     print(pid)
        #     print(batch)

        # print(self.workers.local_worker().policy_map)
        worker = self.workers.local_worker()
        worker.env: MultiAgentEnv
        for policy_id in worker.policy_map.keys():
            print(worker.policy_map[policy_id])
            print(train_batch[policy_id])
        raise NotImplementedError

    @override(A3C)
    @staticmethod
    def _get_env_id_and_creator(env_specifier: Union[str, EnvType, None], config: AlgorithmConfig) -> Tuple[Optional[str], EnvCreator]:
        """This function creats a wrapped environment with central planner"""
        env_id, env_creator = Algorithm._get_env_id_and_creator(env_specifier=env_specifier, config=config)

        return env_id, lambda cfg: MultiAgentEnvWithCentralPlanner(env=env_creator(cfg))
