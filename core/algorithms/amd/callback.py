# typing:
from gymnasium import spaces
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.utils.typing import PolicyID

from .amd_torch_policy import AMDGeneralPolicy
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

    def on_create_policy(self, *, policy_id: PolicyID, policy: AMDGeneralPolicy) -> None:

        # ! This is where we make moficitation on policies
        # ! Let policy object know if it is central planner or not
        policy.is_central_planner = (policy_id == CENTRAL_PLANNER)

        # TODO: change central planner's distribution class so that it can directly pass gradient to model.
