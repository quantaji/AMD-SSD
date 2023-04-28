from ray.rllib.algorithms.callbacks import DefaultCallbacks

# typing:
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.env_context import EnvContext
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import AgentID, EnvType, PolicyID
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch

from .amd_torch_policy import AMDGeneralPolicy
from .constants import CENTRAL_PLANNER, PreLearningProcessing


class AMDDefualtCallback(DefaultCallbacks):

    def on_create_policy(self, *, policy_id: PolicyID, policy: AMDGeneralPolicy) -> None:

        # ! This is where we make moficitation on policies
        # ! Let policy object know if it is central planner or not
        policy.is_central_planner = (policy_id == CENTRAL_PLANNER)

        # TODO: change central planner's distribution class so that it can directly pass gradient to model.

        # raise NotImplementedError
