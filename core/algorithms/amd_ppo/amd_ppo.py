import importlib
import logging
import time
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
from gymnasium import spaces
from gymnasium.vector.utils import batch_space, create_empty_array
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided, Space
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.execution.rollout_ops import standardize_fields, synchronous_parallel_sample
from ray.rllib.execution.train_ops import multi_gpu_train_one_step, train_one_step
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.from_config import NotProvided
from ray.rllib.utils.metrics import NUM_AGENT_STEPS_SAMPLED, NUM_ENV_STEPS_SAMPLED, SYNCH_WORKER_WEIGHTS_TIMER
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.typing import EnvCreator, EnvType, MultiAgentPolicyConfigDict, PolicyID, ResultDict, SampleBatchType
from ray.util.debug import log_once

from ..amd.action_distribution import SigmoidTorchDeterministic, TanhTorchDeterministic
from ..amd.amd import AMD, AMDConfig
from ..amd.callback import AMDDefualtCallback
from ..amd.constants import CENTRAL_PLANNER, SIGMOID_DETERMINISTIC_DISTRIBUTION, TANH_DETERMINISTIC_DISTRIBUTION, PreLearningProcessing
from ..amd.utils import action_to_reward, cumsum_factor_across_eps, get_availability_mask, get_env_example
from ..amd.wrappers import MultiAgentEnvWithCentralPlanner

logger = logging.getLogger(__name__)


class AMDPPOConfig(PPOConfig):

    def __init__(self, algo_class=None):
        super().__init__(algo_class=algo_class or AMDPPO)

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
        self.reward_distribution: str = 'tanh'
        self.awareness_batch_size: int = None
        self.agent_cooperativeness_stats_fn: Callable = None
        self.neural_awareness_method: str = 'grad'

        # some setting for training large models and algorithm, like PPO
        self.use_cum_reward = False
        self.pfactor_half_step = 0
        self.pfactor_step_scale = 1

        ModelCatalog.register_custom_action_dist(TANH_DETERMINISTIC_DISTRIBUTION, TanhTorchDeterministic)
        ModelCatalog.register_custom_action_dist(SIGMOID_DETERMINISTIC_DISTRIBUTION, SigmoidTorchDeterministic)

        self.cp_action_dist_dic = {
            'sigmoid': SIGMOID_DETERMINISTIC_DISTRIBUTION,
            'tanh': TANH_DETERMINISTIC_DISTRIBUTION,
        }

    @override(PPOConfig)
    def callbacks(self, callbacks_class) -> "AMDPPOConfig":
        return AMDConfig.callbacks(self, callbacks_class)

    @override(PPOConfig)
    def environment(self, **kwargs) -> "AMDPPOConfig":
        super().environment(**kwargs)
        env_example = get_env_example(self)
        if (self.coop_agent_list is None) and isinstance(env_example, MultiAgentEnv):
            self.coop_agent_list = list(env_example.get_agent_ids())

        return self

    @override(PPOConfig)
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
        use_cum_reward: Optional[bool] = NotProvided,
        pfactor_half_step: Optional[int] = NotProvided,
        pfactor_step_scale: Optional[int] = NotProvided,
        agent_cooperativeness_stats_fn: Optional[Union[str, Callable]] = NotProvided,
        neural_awareness_method: Optional[str] = NotProvided,
        **kwargs,
    ) -> "AMDPPOConfig":
        """
            use_cum_reward: whether central planner gives reward (True) or gives advantage (False)
            also, at beginning, we don't want central planner to get envovled so much. we set a sigmoid factor with two parameters to control this
                pfactor_half_step: 
                pfactor_step_scale:
            agent_cooperativeness_stats_fn: a function that takes sample batch, and give a stats about the coorperativeness of current batch
            neural_awareness_method: method for calculation of awareness, 'jacboian' is the default method using torch.func.functional_call and torch.func.jacrev. For LSTM, there is bug in current pytorch, therefore, we offer another option 'grad' using torch.autograd.grad, which is might be slow
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

        if use_cum_reward is not NotProvided:
            self.use_cum_reward = use_cum_reward
        if pfactor_half_step is not NotProvided:
            self.pfactor_half_step = pfactor_half_step
        if pfactor_step_scale is not NotProvided:
            self.pfactor_step_scale = pfactor_step_scale

        if cp_model is not NotProvided:
            self.cp_model = cp_model

        if agent_cooperativeness_stats_fn is not NotProvided:
            if isinstance(agent_cooperativeness_stats_fn, str):
                try:
                    self.agent_cooperativeness_stats_fn = importlib.import_module(agent_cooperativeness_stats_fn)
                except:
                    self.agent_cooperativeness_stats_fn = None
            elif callable(agent_cooperativeness_stats_fn):
                self.agent_cooperativeness_stats_fn = agent_cooperativeness_stats_fn

        if neural_awareness_method is not NotProvided:
            assert neural_awareness_method in ['jacobian', 'grad']
            self.neural_awareness_method = neural_awareness_method

        return self

    @override(PPOConfig)
    def get_multi_agent_setup(
        self,
        *,
        policies: MultiAgentPolicyConfigDict | None = None,
        env: EnvType | None = None,
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

        # [optional] not do minibatch sgd for central planner
        policies[CENTRAL_PLANNER].config.sgd_minibatch_size = policies[CENTRAL_PLANNER].config["train_batch_size"]
        policies[CENTRAL_PLANNER].config.num_sgd_iter = 1

        return policies, is_policy_to_train


class AMDPPO(PPO):

    reward_space_unflattened: spaces.Dict
    reward_space: spaces.Box

    @classmethod
    @override(PPO)
    def get_default_config(cls) -> AlgorithmConfig:
        return AMDPPOConfig()

    @classmethod
    @override(PPO)
    def get_default_policy_class(cls, config: AlgorithmConfig) -> Type[Policy] | None:
        if config["framework"] == 'torch':
            from .amd_ppo_torch_policy import AMDPPOAgentTorchPolicy
            return AMDPPOAgentTorchPolicy
        else:
            raise NotImplementedError("Current algorithm only support PyTorch!")

    @override(PPO)
    @staticmethod
    def _get_env_id_and_creator(env_specifier: Union[str, EnvType, None], config: AlgorithmConfig) -> Tuple[Optional[str], EnvCreator]:
        """This function creats a wrapped environment with central planner"""
        return AMD._get_env_id_and_creator(env_specifier=env_specifier, config=config)

    @override(PPO)
    def training_step(self) -> ResultDict:

        # Collect SampleBatches from sample workers until we have a full batch.
        if self.config.count_steps_by == "agent_steps":
            train_batch = synchronous_parallel_sample(
                worker_set=self.workers,
                max_agent_steps=self.config.train_batch_size,
            )
        else:
            train_batch = synchronous_parallel_sample(worker_set=self.workers, max_env_steps=self.config.train_batch_size)
        train_batch = train_batch.as_multi_agent()
        self._counters[NUM_AGENT_STEPS_SAMPLED] += train_batch.agent_steps()
        self._counters[NUM_ENV_STEPS_SAMPLED] += train_batch.env_steps()

        # Standardize advantages
        # NOTE: I am not sure if standardize advange will have some effect on agent!
        # there are notes saying this will not affect results much, so I will keep it here
        train_batch = standardize_fields(train_batch, ["advantages"])

        # NOTE: this is the start of my own code
        # for conviniency
        worker = self.workers.local_worker()
        policy_map = worker.policy_map

        # update appearance
        policy_map[CENTRAL_PLANNER].appearance = self.appearance

        # ! ALL the Pre-learning processing happens here
        if train_batch.agent_steps() > 0 and self.config['planner_reward_max'] > 0.0:
            train_batch = self.prelearning_process_trajectory(train_batch)
        # NOTE: this is the end of my own code

        # Train
        train_results = {}
        if train_batch.agent_steps() > 0:
            if self.config._enable_rl_trainer_api:
                train_results = self.trainer_runner.update(train_batch)
            elif self.config.simple_optimizer:
                train_results = train_one_step(self, train_batch)
            else:
                train_results = multi_gpu_train_one_step(self, train_batch)

            policies_to_update = list(train_results.keys())
        else:
            time.sleep(1)

        global_vars = {
            "timestep": self._counters[NUM_AGENT_STEPS_SAMPLED],
            "num_grad_updates_per_policy": {
                pid: self.workers.local_worker().policy_map[pid].num_grad_updates
                for pid in policies_to_update
            },
        }

        # Update weights - after learning on the local worker - on all remote
        # workers.
        if self.workers.num_remote_workers() > 0:
            with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
                from_worker = None
                if self.config._enable_rl_trainer_api:
                    from_worker = self.trainer_runner
                self.workers.sync_weights(
                    from_worker=from_worker,
                    policies=list(train_results.keys()),
                    global_vars=global_vars,
                )

        if self.config._enable_rl_trainer_api:
            kl_dict = {pid: pinfo[LEARNER_STATS_KEY].get("kl") for pid, pinfo in train_results.items()}
            # triggers a special update method on RLOptimizer to update the KL values.
            self.trainer_runner.additional_update(kl_values=kl_dict)

            return train_results

        # For each policy: Update KL scale and warn about possible issues
        for policy_id, policy_info in train_results.items():

            if policy_id == CENTRAL_PLANNER:
                continue

            # Update KL loss with dynamic scaling
            # for each (possibly multiagent) policy we are training
            kl_divergence = policy_info[LEARNER_STATS_KEY].get("kl")
            self.get_policy(policy_id).update_kl(kl_divergence)

            # Warn about excessively high value function loss
            scaled_vf_loss = (self.config.vf_loss_coeff * policy_info[LEARNER_STATS_KEY]["vf_loss"])
            policy_loss = policy_info[LEARNER_STATS_KEY]["policy_loss"]
            if (log_once("ppo_warned_lr_ratio") and self.config.get("model", {}).get("vf_share_layers") and scaled_vf_loss > 100):
                logger.warning("The magnitude of your value function loss for policy: {} is "
                               "extremely large ({}) compared to the policy loss ({}). This "
                               "can prevent the policy from learning. Consider scaling down "
                               "the VF loss by reducing vf_loss_coeff, or disabling "
                               "vf_share_layers.".format(policy_id, scaled_vf_loss, policy_loss))
            # Warn about bad clipping configs.
            train_batch.policy_batches[policy_id].set_get_interceptor(None)
            mean_reward = train_batch.policy_batches[policy_id]["rewards"].mean()
            if (log_once("ppo_warned_vf_clip") and mean_reward > self.config.vf_clip_param):
                self.warned_vf_clip = True
                logger.warning(f"The mean reward returned from the environment is {mean_reward}"
                               f" but the vf_clip_param is set to {self.config['vf_clip_param']}."
                               f" Consider increasing it for policy: {policy_id} to improve"
                               " value function convergence.")

        # Update global vars on local worker as well.
        self.workers.local_worker().set_global_vars(global_vars)

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
                batch[PreLearningProcessing.AVAILABILITY] = avail

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

        # ! options 1: assume the central planner delivers reward, so we have so sum all rewards in an epislode
        # ! options 2: we can assume the planner delivers advantages
        if self.config['use_cum_reward']:
            factor_matrix = cumsum_factor_across_eps(cp_eps_id)
            # make accumulation: R_{\tau} = \sum_{t=\tau}^{END} r_t
            r_planner_cum = np.repeat(
                factor_matrix @ r_planner,
                factor_matrix.sum(axis=-1),
                axis=0,
            )  # (T, n_agents)
            # this step is used for checking consistence on sampling and training
        else:
            r_planner_cum = r_planner
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

        train_batch[CENTRAL_PLANNER][PreLearningProcessing.TOTAL_ADVANTAGES] = total_advantages

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
