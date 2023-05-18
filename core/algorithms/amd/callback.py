import numpy as np
from gymnasium import spaces
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.utils.typing import PolicyID

from .constants import CENTRAL_PLANNER


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

        # get a mask of array indicating which agent are in compuation
        appearance_mask_unflattened = {}  # stores which agent shows up in dict, to minus its mean
        for agent_id in algorithm.reward_space_unflattened.keys():
            appearance_mask_unflattened[agent_id] = (agent_id in algorithm.config['coop_agent_list'] or (algorithm.config['coop_agent_list'] is None))
        algorithm.appearance: np.ndarray = spaces.flatten(algorithm.reward_space_unflattened, appearance_mask_unflattened).astype(bool)  # shape of (n_agent, )
        # remember to send it to central planner's policy when doing a training step

    def on_create_policy(self, *, policy_id: PolicyID, policy: TorchPolicyV2) -> None:

        # ! This is where we make moficitation on policies
        # ! Let policy object know if it is central planner or not
        policy.is_central_planner = (policy_id == CENTRAL_PLANNER)
