from ray.rllib.env import MultiAgentEnv
from pettingzoo.utils.env import AECEnv, ParallelEnv
from pettingzoo.sisl import waterworld_v4
from ray.rllib.env import PettingZooEnv
from typing import Callable, Dict, List, Tuple, Optional, Union, Set, Type
from ray.rllib.utils.typing import (
    AgentID,
    EnvCreator,
    EnvID,
    EnvType,
    MultiAgentDict,
    MultiEnvDict,
)
import gymnasium as gym
from gymnasium import spaces
from pettingzoo.butterfly import cooperative_pong_v5
from pettingzoo.atari import maze_craze_v3
from pettingzoo.test import api_test



# class WolfpackEnv(MultiAgentEnv):

#     def __init__(self):

#         a = spaces.Dict
#         raise NotImplementedError
#         super().__init__()

#         # TODO: need to implement
#         # self.observation_space: gym.spaces.Dict
#         # self.action_space: gym.spaces.Dict
#         # self._agent_ids

#     def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[MultiAgentDict, MultiAgentDict]:
#         raise NotImplementedError
#         return super().reset(seed=seed, options=options)

#     def step(self, action_dict: MultiAgentDict) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
#         raise NotImplementedError
#         return super().step(action_dict)

#     def render(self) -> None:
#         raise NotImplementedError
#         return super().render()

#     def with_agent_groups(self, groups: Dict[str, List[AgentID]], obs_space: gym.Space = None, act_space: gym.Space = None) -> "MultiAgentEnv":
#         raise NotImplementedError
#         return super().with_agent_groups(groups, obs_space, act_space)


# class Config(object):
#     def __init__(self):
#         super(Config, self).__init__()

#         self._set_action_dict()
#         self._set_orientation_dict()
#         self._set_grid_dict()
#         self._set_color_dict()

#     def _set_action_dict(self):
#         self.action_dict = {
#             "stay": np.array([0, 0]),
#             "move_up": np.array([-1, 0]),
#             "move_down": np.array([1, 0]),
#             "move_right": np.array([0, 1]),
#             "move_left": np.array([+0, -1]),
#             "spin_right": +1,
#             "spin_left": -1,
#         }

#     def _set_orientation_dict(self):
#         self.orientation_dict = {
#             "up": 0,
#             "right": 1,
#             "down": 2,
#             "left": 3,
#         }

#         self.orientation_delta_dict = {
#             "up": np.array([-1, 0]),
#             "right": np.array([0, 1]),
#             "down": np.array([1, 0]),
#             "left": np.array([+0, -1]),
#         }

#     def _set_grid_dict(self):
#         self.grid_dict = {
#             "empty": 0,
#             "wall": 1,
#             "prey": 2,
#             "predator": 3,
#             "orientation": 4,
#         }

#     def _set_color_dict(self):
#         self.color_dict = {
#             "empty": [0., 0., 0.],  # Black
#             "wall": [0.5, 0.5, 0.5],  # Gray
#             "prey": [1., 0., 0.],  # Red
#             "predator": [0., 0., 1.],  # Blue
#             "other_predator": [0., 0., 0.5],  # Light Blue
#             "orientation": [0.1, 0.1, 0.1],  # Dark Gray
#         }

WOLFPACK_MAP = [
    '@@@@@@@@@@@@@@@@@@@@',
    '@                  @',
    '@      @@    @     @',
    '@@@     @@    @    @',
    '@                  @',
    '@                @ @',
    '@       @@      @  @',
    '@ @@@          @   @',
    '@ @@@              @',
    '@         @@@      @',
    '@    @     @   @   @',
    '@   @@        @@   @',
    '@    @@      @@    @',
    '@                  @',
    '@           @     @@',
    '@   @@     @@@    @@',
    '@  @@@      @@@   @@',
    '@   @              @',
    '@           @      @',
    '@@@@@@@@@@@@@@@@@@@@',
]



# from pettingzoo.utils.env import ParallelEnv
# from pettingzoo.utils.env import ObsType, ActionType, AgentID, ObsDict, ActionDict
# from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, TypeVar
# import gymnasium.spaces as space
# import numpy as np


# class WolfpackEnv(ParallelEnv):

#     possible_agents = ['wolf_1', 'wolf_2', 'prey']

#     # all three players have same actions, the following list is only for illustration, not for usage.
#     action_list = [
#         "stay",
#         "move_up",
#         "move_down",
#         "move_right",
#         "move_left",
#         "spin_right",
#         "spin_left",
#     ]
#     action_spaces = {
#         'wolf_1': space.Discrete(6),
#         'wolf_2': space.Discrete(6),
#         'prey': space.Discrete(6),
#     }

#     def __init__(self):
#         super().__init__()

#     def step(self, action: ActionType) -> None:
#         return super().step(action)
