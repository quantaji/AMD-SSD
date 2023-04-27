from __future__ import annotations

import warnings
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, TypeVar

from gymnasium.vector.utils import create_empty_array
import numpy as np
from numpy import ndarray
from gymnasium.spaces import Space
from gymnasium import spaces

from pettingzoo.utils.env import ParallelEnv
from pettingzoo.utils.env import ObsType, ActionType, AgentID, ObsDict, ActionDict

from ray.rllib.env.multi_agent_env import MultiAgentEnv, ENV_STATE

from ..amd import PreLearningProcessing

STATE_SPACE = 'state_space'


class MultiAgentEnvFromPettingZooParallel(MultiAgentEnv):

    def __init__(self, env: ParallelEnv):

        self.par_env = env
        self.par_env.reset()

        self.observation_space = spaces.Dict(self.par_env.observation_spaces)
        self.action_space = spaces.Dict(self.par_env.observation_spaces)
        self._agent_ids = set(self.par_env.possible_agents)

        # see if state is callable
        try:
            self.par_env.state()
            self.state = self.par_env.state
        except:
            pass

        # see if it specifies state space
        if hasattr(self.par_env, STATE_SPACE) and isinstance(getattr(self.par_env, STATE_SPACE), Space):
            self.state_space = getattr(self.par_env, STATE_SPACE)

        super().__init__()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = self.par_env.reset(seed=seed, return_info=True, options=options)
        return obs, info or {}

    def step(self, action_dict):
        obss, rews, terminateds, truncateds, infos = self.par_env.step(action_dict)
        terminateds["__all__"] = all(terminateds.values())
        truncateds["__all__"] = all(truncateds.values())
        return obss, rews, terminateds, truncateds, infos

    def close(self):
        self.par_env.close()

    def render(self):
        return self.par_env.render()

    @property
    def get_sub_environments(self):
        return self.par_env.unwrapped
