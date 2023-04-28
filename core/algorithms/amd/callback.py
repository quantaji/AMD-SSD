from ray.rllib.algorithms.callbacks import DefaultCallbacks

# typing:
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.env_context import EnvContext
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import AgentID, EnvType, PolicyID
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch

from gymnasium import spaces

from .amd_torch_policy import AMDGeneralPolicy
from .constants import CENTRAL_PLANNER, PreLearningProcessing


class AMDDefualtCallback(DefaultCallbacks):

    def on_algorithm_init(self, *, algorithm: Algorithm, **kwargs) -> None:
        # create a example env for referencing
        reward_space_unflattened = {}
        policy_map = algorithm.workers.local_worker().policy_map
        for policy_id in policy_map.keys():
            if policy_id is not CENTRAL_PLANNER:
                reward_space_unflattened[policy_id] = spaces.Box(-1, 1)
        algorithm.reward_space_unflattened = spaces.Dict(reward_space_unflattened)
        algorithm.reward_space = spaces.flatten_space(algorithm.reward_space_unflattened)

    def on_create_policy(self, *, policy_id: PolicyID, policy: AMDGeneralPolicy) -> None:

        # ! This is where we make moficitation on policies
        # ! Let policy object know if it is central planner or not
        policy.is_central_planner = (policy_id == CENTRAL_PLANNER)

        # TODO: change central planner's distribution class so that it can directly pass gradient to model.

        # raise NotImplementedError


# class AMDAgentWarenessCallback(DefaultCallbacks):
#     """
#     This callback:
#         1. generate the single agents' awareness. (sum_t V_t g_log_p_t)^T(sum_t g_log_p_t), of shape (T, ).
#         2. it generate the SampleBatch needed for the central planner.
#         3. forward these to central planner and get the planner's rewards r_planner, and update each single agent's sample batch
#     """

#     def on_sample_end(self, *, worker: RolloutWorker, samples: SampleBatch, **kwargs) -> None:

#         # # ! STEP 1: get each agent's awareness at each timestep
#         # if isinstance(samples, MultiAgentBatch):
#         #     for pid, batch in samples.policy_batches.items():
#         #         batch.decompress_if_needed()
#         #         policy: AMDAgentPolicy = worker.policy_map[pid]
#         #         print('-----*****-----', pid, batch, batch[SampleBatch.T].dtype, batch[SampleBatch.EPS_ID].dtype)
#         #         policy.config['policy_param'] = 'neural'
#         #         raise NotImplementedError
#         # else:
#         #     raise ValueError("This should be a multi-agent sample batch, check your algorithm configuration and environment!")
#         # # print(type(samples), samples, worker.policy_dict, worker.policy_map, worker.policy_config)

#         # MultiAgentBatch({'wolf_1': SampleBatch(128: ['obs', 'actions', 'rewards', 'terminateds', 'truncateds', 'infos', 'eps_id', 'unroll_id', 'agent_index', 't', 'advantages', 'value_targets']),

#         pass

#         # TODO ! STEP 2: Generate obs for central planner, and get each agent's awareness as a dict.

#         # raise NotImplementedError

#         # TODO ! STEP 3: update each single agent's sample batch
