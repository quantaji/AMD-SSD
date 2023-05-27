import time
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
from gymnasium import spaces
from gymnasium.vector.utils import batch_space, create_empty_array
from ray.rllib.algorithms.a3c import A3C, A3CConfig
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import (AlgorithmConfig, NotProvided, Space)
from ray.rllib.algorithms.callbacks import DefaultCallbacks, MultiCallbacks
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.execution.train_ops import (multi_gpu_train_one_step, train_one_step)
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_action_dist import TorchDeterministic
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.from_config import NotProvided
from ray.rllib.utils.metrics import (NUM_AGENT_STEPS_SAMPLED, NUM_ENV_STEPS_SAMPLED, SYNCH_WORKER_WEIGHTS_TIMER)
from ray.rllib.utils.typing import (EnvCreator, EnvType, MultiAgentPolicyConfigDict, PolicyID, ResultDict, SampleBatchType)

from .action_distribution import TanhTorchDeterministic, SigmoidTorchDeterministic
from .callback import AMDDefualtCallback
from .constants import (CENTRAL_PLANNER, TANH_DETERMINISTIC_DISTRIBUTION, SIGMOID_DETERMINISTIC_DISTRIBUTION, PreLearningProcessing)
from .utils import (action_to_reward, discounted_cumsum_factor_matrix, get_availability_mask, get_env_example)
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
        self.agent_pseudo_lr = 1.0
        self.central_planner_lr = None
        self.cp_model = None
        self.coop_agent_list = None
        self.param_assumption: str = 'neural'
        self.planner_reward_max: float = 1.0
        self.force_zero_sum: bool = False
        self.planner_reward_cost: float = 0.0
        self.reward_distribution: str = 'sigmoid'
        self.awareness_batch_size: int = None

        # we don't sample async so
        self.sample_async = False

        ModelCatalog.register_custom_action_dist(TANH_DETERMINISTIC_DISTRIBUTION, TanhTorchDeterministic)
        ModelCatalog.register_custom_action_dist(SIGMOID_DETERMINISTIC_DISTRIBUTION, SigmoidTorchDeterministic)

        self.cp_action_dist_dic = {
            'sigmoid': SIGMOID_DETERMINISTIC_DISTRIBUTION,
            'tanh': TANH_DETERMINISTIC_DISTRIBUTION,
        }

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
        cp_model: Optional[dict] = NotProvided,
        coop_agent_list: Optional[List[str]] = NotProvided,
        param_assumption: Optional[str] = NotProvided,
        planner_reward_max: Optional[float] = NotProvided,
        force_zero_sum: Optional[bool] = NotProvided,
        planner_reward_cost: Optional[float] = NotProvided,
        reward_distribution: Optional[str] = NotProvided,
        awareness_batch_size: Optional[int] = NotProvided,
        **kwargs,
    ) -> "AMDConfig":
        """
        agent_pseudo_lr: the learning rate of individual agents assumed by central planner, default is 1.0.
        central_planner_lr: the learning rate of central planner, default to be lr TODO currently this is not supported
        coop_agent_list: a list of agent_id in the environment that is wished to be designed to be cooperative. For example, when agent includes prey and predator, we only wish preditors to be cooperative. This spetifies the agent in amd's loss.
        param_assumption: assump the policy to be parametrized by direct softmax or neural, this affect how the algoirthm calculate the awareness.
        planner_reward_max: [-R, R] for planner
        force_zero_sum: force the reward to minus its mean, in all cooperative agents.
        planner_reward_cost: the loss penalty of using reward
        reward_distribution: diterministic, whether tanh or sigmoid
        awareness_batch_size: the batch size for computing awareness, used for saving gpu vram, default is None, which means all the batch
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
        if planner_reward_cost is not NotProvided:
            self.planner_reward_cost = planner_reward_cost

        assert reward_distribution in ['sigmoid', 'tanh', NotProvided], "reward_distribution should be either 'tanh' or 'sigmoid'!"
        if reward_distribution is not NotProvided:
            self.reward_distribution = reward_distribution

        if awareness_batch_size is not NotProvided:
            self.awareness_batch_size = awareness_batch_size

        if cp_model is not NotProvided:
            self.cp_model = cp_model

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

        # here, modify central planner's distribution
        # then in rollout worker, it will create the corresponding policy with distribution
        # the policies now is a dict that maps to algorithm config object
        if self.central_planner_lr:
            policies[CENTRAL_PLANNER].config.lr = self.central_planner_lr

        if self.cp_model:
            for key in self.cp_model.keys():
                policies[CENTRAL_PLANNER].config.model[key] = self.cp_model[key]

        policies[CENTRAL_PLANNER].config.model['custom_action_dist'] = self.cp_action_dist_dic[self.reward_distribution]

        # force central planner to not explore
        policies[CENTRAL_PLANNER].config.explore = False

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

        # update appearance
        policy_map[CENTRAL_PLANNER].appearance = self.appearance

        # ! ALL the Pre-learning processing happens here
        if train_batch.agent_steps() > 0:
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

    def prelearning_process_trajectory(self, train_batch: SampleBatch) -> SampleBatch:

        # ! STEP 0: useful information
        batch_size = train_batch[CENTRAL_PLANNER][SampleBatch.ACTIONS].shape[0]
        batch_uf = batch_space(self.reward_space_unflattened, batch_size)
        ref_vector = create_empty_array(self.reward_space, batch_size)  # this is a reference vector of shape (T, n_agents)
        policy_map = self.workers.local_worker().policy_map

        # ! STEP 1: get central planer's t, episode_id, calculate factor matrix
        cp_t = train_batch[CENTRAL_PLANNER][SampleBatch.T]
        cp_eps_id = train_batch[CENTRAL_PLANNER][SampleBatch.EPS_ID]

        cumsum_factor_matrix = discounted_cumsum_factor_matrix(eps_id=cp_eps_id, t=cp_t)
        train_batch[CENTRAL_PLANNER][PreLearningProcessing.DISCOUNTED_FACTOR_MATRIX] = cumsum_factor_matrix

        # ! STEP 2: get availability
        avail_uf = create_empty_array(self.reward_space_unflattened, batch_size)
        for policy_id in self.config["coop_agent_list"]:
            if policy_id in policy_map.keys():
                batch = train_batch[policy_id]
                # compute availability
                avail = get_availability_mask(
                    cp_t=cp_t,
                    cp_eps_id=cp_eps_id,
                    ag_t=batch[SampleBatch.T],
                    ag_eps_id=batch[SampleBatch.EPS_ID],
                )  # (T, )
                # modify availability
                avail_uf[policy_id] = avail.reshape(-1, 1).astype(float)  # (T, 1)
        availability = spaces.flatten(batch_uf, avail_uf).reshape(ref_vector.T.shape).T.astype(bool)  # (T, n_agents)
        train_batch[CENTRAL_PLANNER][PreLearningProcessing.AVAILABILITY] = availability

        # ! STEP 3: get reward by planner
        # note that rllib may have exploration configuration, we don't what this to happen to central planner, therefore, we forward central planner again
        cp_policy = policy_map[CENTRAL_PLANNER]
        actions = cp_policy.compute_central_planner_actions(train_batch[CENTRAL_PLANNER])
        train_batch[CENTRAL_PLANNER][SampleBatch.ACTIONS] = actions

        r_planner: np.ndarray = action_to_reward(
            actions=actions,
            availability=availability,
            appearance=self.appearance,
            reward_max=self.config['planner_reward_max'],
            zero_sum=self.config['force_zero_sum'],
        )
        train_batch[CENTRAL_PLANNER][PreLearningProcessing.R_PLANNER] = r_planner  # (T, n_agents)

        # make accumulation: R_{\tau} = \sum_{t=\tau}^{END} r_t
        r_planner_cum = cumsum_factor_matrix @ r_planner  # (T, n_agents)
        # this step is used for checking consistence on sampling and training
        train_batch[CENTRAL_PLANNER][PreLearningProcessing.R_PLANNER_CUM] = r_planner_cum
        r_planner_cum_uf = spaces.unflatten(batch_uf, r_planner_cum.T.flatten())  # this transpose is very important!, each of shape (T, 1)

        # distribute accumulated reward to each agent
        for policy_id in self.config["coop_agent_list"]:
            if policy_id in policy_map.keys():
                avail = avail_uf[policy_id].reshape(-1).astype(bool)
                train_batch[policy_id][PreLearningProcessing.R_PLANNER_CUM] = r_planner_cum_uf[policy_id][avail, :].reshape(-1)  # (T, 1)

        # ! STEP 4: calculate the sum of all advantages of cooperative agents
        total_advantages = np.zeros((batch_size, ))  # placeholder
        for policy_id in self.config["coop_agent_list"]:
            if policy_id in policy_map.keys():
                batch = train_batch[policy_id]
                avail = avail_uf[policy_id].reshape(-1).astype(bool)
                total_advantages[avail] += batch[Postprocessing.ADVANTAGES]
                # total_advantages[avail] += batch[SampleBatch.REWARDS]

        train_batch[CENTRAL_PLANNER][PreLearningProcessing.TOTAL_ADVANTAGES] = total_advantages
        # train_batch[CENTRAL_PLANNER][PreLearningProcessing.TOTAL_ADVANTAGES] = cumsum_factor_matrix @ total_advantages

        # then distribute them to all cooperative agents
        for policy_id in self.config["coop_agent_list"]:
            if policy_id in policy_map.keys():
                avail = avail_uf[policy_id].reshape(-1).astype(bool)
                train_batch[policy_id][PreLearningProcessing.TOTAL_ADVANTAGES] = total_advantages[avail]

        # ! STEP 5: computing awareness for each agent
        awareness_uf = create_empty_array(self.reward_space_unflattened, batch_size)
        for policy_id in self.config["coop_agent_list"]:
            if policy_id in policy_map.keys():
                batch = train_batch[policy_id]
                batch = policy_map[policy_id].compute_awareness(batch)
                # aggregate to central planner
                avail = avail_uf[policy_id].reshape(-1).astype(bool)

                awareness_uf[policy_id][avail, :] = batch[PreLearningProcessing.AWARENESS].reshape(-1, 1)
        awareness = spaces.flatten(batch_uf, awareness_uf).reshape(ref_vector.T.shape).T  # the transpose is necessary
        train_batch[CENTRAL_PLANNER][PreLearningProcessing.AWARENESS] = awareness

        return train_batch

    @override(A3C)
    @staticmethod
    def _get_env_id_and_creator(env_specifier: Union[str, EnvType, None], config: AlgorithmConfig) -> Tuple[Optional[str], EnvCreator]:
        """This function creats a wrapped environment with central planner"""
        env_id, env_creator = Algorithm._get_env_id_and_creator(env_specifier=env_specifier, config=config)

        return env_id, lambda cfg: MultiAgentEnvWithCentralPlanner(env=env_creator(cfg))
