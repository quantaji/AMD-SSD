from __future__ import annotations

import warnings
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, TypeVar
from gymnasium.core import Env

from gymnasium import spaces
from gymnasium.vector.utils import create_empty_array
import numpy as np
from numpy import ndarray
from gymnasium.spaces import Space

from pettingzoo.utils.env import ParallelEnv
from pettingzoo.utils.env import ObsType, ActionType, AgentID, ObsDict, ActionDict

from ray.rllib.env.multi_agent_env import MultiAgentEnv, ENV_STATE
from ray.rllib.utils.typing import MultiAgentDict

from copy import deepcopy

from ..amd import PreLearningProcessing

STATE_SPACE = 'state_space'


class MultiAgentEnvWithCentralPlanner(MultiAgentEnv):

    CENTRAL_PLANNER: str = 'central_planner'
    observation_space: Optional[spaces.Dict] = None
    action_space: Optional[spaces.Dict] = None
    _agent_ids: set[str]
    _individual_agent_ids: set[str]
    central_planner_state_space_unflattend: Space

    def __init__(self, env: MultiAgentEnv):

        self.env: MultiAgentEnv = env
        self.env.reset()

        # make sure central planner's id is correct
        while self.CENTRAL_PLANNER in self.env.get_agent_ids():
            self.CENTRAL_PLANNER = self.CENTRAL_PLANNER + '_'

        # try if the env have .state() function, if so, the central planner takes the sate, otherwise it returns a flattened dict
        self.have_state = hasattr(self.env, STATE_SPACE) and hasattr(self.env, ENV_STATE) and callable(getattr(self.env, ENV_STATE))
        if self.have_state:
            try:
                getattr(self.env, ENV_STATE)()
                self.state = getattr(self.env, ENV_STATE)
                self.state_space = getattr(self.env, STATE_SPACE)
            except:
                self.have_state = False

        # copy variables from original env
        self._agent_ids = deepcopy(self.env.get_agent_ids())
        self.observation_space = deepcopy(spaces.Dict(self.env.observation_space))
        self.action_space = deepcopy(spaces.Dict(self.env.action_space))

        # register central planner
        self._individual_agent_ids = deepcopy(set(self._agent_ids))
        self._agent_ids.add(self.CENTRAL_PLANNER)

        # add observation space
        if self.have_state:
            self.central_planner_state_space_unflattend = getattr(self.env, STATE_SPACE)
            self.observation_space[self.CENTRAL_PLANNER] = getattr(self.env, STATE_SPACE)
        else:
            self.central_planner_state_space_unflattend = spaces.Dict(self.observation_space)
            self.observation_space[self.CENTRAL_PLANNER] = spaces.flatten_space(self.central_planner_state_space_unflattend)

        super().__init__()

    def observe_central_planner(self, obss: MultiAgentDict):
        if self.have_state:
            return self.state()
        else:
            obs_cp = {}
            for agent_id in self.env.get_agent_ids():
                if agent_id in obss.keys():
                    obs_cp[agent_id] = obss[agent_id]
                else:
                    obs_cp[agent_id] = create_empty_array(self.observation_space[agent_id])

            return spaces.flatten(self.central_planner_state_space_unflattend, obs_cp)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> Tuple[MultiAgentDict, MultiAgentDict]:
        obss, infos = self.env.reset(seed=seed, options=options)

        # now get the observation of central planner
        obss[self.CENTRAL_PLANNER] = self.observe_central_planner(obss)

        return obss, infos

    def step(self, action_dict: MultiAgentDict) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:

        obss, rews, terminateds, truncateds, infos = self.env.step(action_dict)

        terminateds[self.CENTRAL_PLANNER] = terminateds["__all__"]
        truncateds[self.CENTRAL_PLANNER] = truncateds["__all__"]
        infos[self.CENTRAL_PLANNER] = {}

        rews[self.CENTRAL_PLANNER] = 0.0

        obss[self.CENTRAL_PLANNER] = self.observe_central_planner(obss)

        return obss, rews, terminateds, truncateds, infos

    def render(self) -> None:
        return self.env.render()

    def close(self):
        return self.env.close()

    def unwrapped(self) -> Env:
        return self.env
