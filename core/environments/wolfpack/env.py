from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
)
from pettingzoo.utils.env import (
    ObsType,
    ActionType,
    AgentID,
    ObsDict,
    ActionDict,
)

import pygame
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from pettingzoo.utils.env import ParallelEnv

from ..base.gridworld import GridWorldBase
from .agent import WolfpackAgent
from .constants import (
    WOLFPACK_MAP,
    WOLFPACK_STATE,
    WOLFPACK_STATE_SHAPE,
    WOLFPACK_COLOR,
    WOLFPACK_ORIENTATION_TUNE,
    WOLFPACK_NO_ENTRY_STATE,
    WOLFPACK_ACTIONS,
    WOLFPACK_OBSERVATION_SHAPE,
    WOLFPACK_AGENT_MAP,
    WOLFPACK_ORIENTATION_BOUNDING_BOX,
    WOLFPACK_AGENT_VIEW_TUNE,
)
from core.environments.utils import ascii_list_to_array, ascii_array_to_rgb_array, ascii_dict_to_color_array


class Wolfpack(GridWorldBase):

    base_world = ascii_list_to_array(WOLFPACK_MAP)
    world_shape = base_world.shape
    ascii_color_dict = WOLFPACK_COLOR
    ascii_color_array = ascii_dict_to_color_array(ascii_color_dict)

    resolution = [640, 640]
    block_size_h, block_size_w = 32, 32
    line_width = 0

    agents = ['wolf_1', 'wolf_2', 'prey']

    CAPTURE_RADIUS = 6

    def __init__(
        self,
        randomizer: np.random.Generator,
        r_lone: float = 1.0,
        r_team: float = 5.0,
        r_prey: float = 0.1,  # living reward for prey, this is constantly given
        max_cycles: int = 1024,
    ):

        self.agent_dict = {}
        for agent in self.agents:
            self.agent_dict[agent] = WolfpackAgent(agent)

        self.r_lone = r_lone
        self.r_team = r_team
        self.r_prey = r_prey

        self.max_cycles = max_cycles

        self.randomizer = randomizer

        super().__init__()

    def reset(self):
        # reset seed

        # reset agents
        x, y = np.where(~np.isin(self.base_world, WOLFPACK_NO_ENTRY_STATE))

        pos_choice = self.randomizer.choice(len(x), size=(len(self.agents), ), replace=False)
        ori_choice = self.randomizer.choice(4, size=(len(self.agents), ), replace=True)
        for i, agent in enumerate(self.agents):
            self.agent_dict[agent].position = [x[pos_choice[i]], y[pos_choice[i]]]
            self.agent_dict[agent].orientation = ori_choice[i]

        # reset counters, etc
        self.reinit()

        self.num_frames = 0

        # update drawing
        self.draw()

    def grid_world(self) -> np.ndarray:
        # load base world
        grid_world = self.base_world.copy()

        # add agent position
        for agent_id in self.agent_dict.keys():
            posi = self.agent_dict[agent_id].position
            if grid_world[posi[0], posi[1]] == ' ':
                grid_world[posi[0], posi[1]] = WOLFPACK_AGENT_MAP[agent_id]
            else:
                grid_world[posi[0], posi[1]] = 'W'

        # add orientation
        for agent_id in self.agent_dict.keys():
            ori_posi = self.agent_dict[agent_id].orientation_position
            if (0 <= ori_posi[0]) and (ori_posi[0] < self.world_shape[0]) and (0 <= ori_posi[1]) and (ori_posi[1] < self.world_shape[1]) and grid_world[ori_posi[0], ori_posi[1]] in WOLFPACK_ORIENTATION_TUNE.keys():
                grid_world[ori_posi[0], ori_posi[1]] = WOLFPACK_ORIENTATION_TUNE[grid_world[ori_posi[0], ori_posi[1]]]

        return grid_world

    def observe(self, agent: AgentID) -> spaces.Space:
        # load base world
        grid_world = self.base_world.copy()

        # add agent
        for other_agent in self.agent_dict.keys():
            posi = self.agent_dict[other_agent].position
            if grid_world[posi[0], posi[1]] == ' ':
                grid_world[posi[0], posi[1]] = WOLFPACK_AGENT_VIEW_TUNE[agent][WOLFPACK_AGENT_MAP[other_agent]]
            else:
                grid_world[posi[0], posi[1]] = 'W'

        # add orientation
        for agent_id in self.agent_dict.keys():
            ori_posi = self.agent_dict[agent_id].orientation_position
            if (0 <= ori_posi[0]) and (ori_posi[0] < self.world_shape[0]) and (0 <= ori_posi[1]) and (ori_posi[1] < self.world_shape[1]) and grid_world[ori_posi[0], ori_posi[1]] in WOLFPACK_ORIENTATION_TUNE.keys():
                grid_world[ori_posi[0], ori_posi[1]] = WOLFPACK_ORIENTATION_TUNE[grid_world[ori_posi[0], ori_posi[1]]]

        offset = 20
        grid_world = np.pad(grid_world, pad_width=((offset, offset), (offset, offset)), mode='constant', constant_values='0')

        posi = self.agent_dict[agent].position
        ori = self.agent_dict[agent].orientation

        # x_min, x_max, y_min, y_max = np.array() + offset
        x_min = offset + WOLFPACK_ORIENTATION_BOUNDING_BOX[ori][0] + posi[0]
        x_max = offset + WOLFPACK_ORIENTATION_BOUNDING_BOX[ori][1] + posi[0]
        y_min = offset + WOLFPACK_ORIENTATION_BOUNDING_BOX[ori][2] + posi[1]
        y_max = offset + WOLFPACK_ORIENTATION_BOUNDING_BOX[ori][3] + posi[1]

        observation = grid_world[x_min:x_max + 1, y_min:y_max + 1]
        observation = np.rot90(observation, k=ori)

        return ascii_array_to_rgb_array(observation, self.ascii_color_array)

    def step(self, actions: ActionDict):
        """This function only calculates make action, test ending, and calculate reward
        """

        # act all agent first
        for agent in self.agent_dict.keys():
            self.agent_dict[agent].act(
                action=actions[agent],
                grid_world=self.base_world,
            )  # wolf and prey may went into sampe place, this means termination, two wolf can be at same place

        # counter
        self.num_frames += 1

        # test termination and calculate reward
        dist_1 = np.linalg.norm(self.agent_dict['wolf_1'].position - self.agent_dict['prey'].position, ord=1)
        dist_2 = np.linalg.norm(self.agent_dict['wolf_2'].position - self.agent_dict['prey'].position, ord=1)

        self.reinit()  # reset all rewards etc
        self.rewards['prey'] = self.r_prey  # give living rewards for prey

        if (dist_1 <= 1.0) or (dist_2 <= 1.0):
            # terminates
            self.terminations = dict(zip(self.agents, [True] * len(self.agents)))

            if (dist_1 <= self.CAPTURE_RADIUS) and (dist_2 <= self.CAPTURE_RADIUS):  # cooperative success
                self.rewards['wolf_1'] = self.r_team
                self.rewards['wolf_2'] = self.r_team
            else:  # non-cooperative
                if (dist_1 <= self.CAPTURE_RADIUS):
                    self.rewards['wolf_1'] = self.r_lone
                if (dist_2 <= self.CAPTURE_RADIUS):
                    self.rewards['wolf_2'] = self.r_lone

        elif self.num_frames >= self.max_cycles:
            self.truncations = dict(zip(self.agents, [True] * len(self.agents)))

        if self.render_on:
            pygame.event.pump()
        self.draw()


class WolfpackEnv(ParallelEnv):

    metadata = {
        "name": "wolfpack",
        "render_modes": ["human", "rgb_array", "ansi"],
    }

    observation_shape = (WOLFPACK_OBSERVATION_SHAPE[0], WOLFPACK_OBSERVATION_SHAPE[1], 3)
    state_shape = (WOLFPACK_STATE_SHAPE[0], WOLFPACK_STATE_SHAPE[1], 3)

    def __init__(self, **kwargs) -> None:

        self._kwargs = kwargs

        self.env: Wolfpack

        self.seed()

        self.render_mode = self.env.render_mode
        self.possible_agents = self.env.agents[:]
        self.agents = self.env.agents[:]

        # spaces
        self.observation_spaces = {
            'wolf_1': spaces.Box(low=0, high=255, shape=self.observation_shape, dtype=np.uint8),
            'wolf_2': spaces.Box(low=0, high=255, shape=self.observation_shape, dtype=np.uint8),
            'prey': spaces.Box(low=0, high=255, shape=self.observation_shape, dtype=np.uint8),
        }
        self.action_spaces = {
            'wolf_1': spaces.Discrete(len(WOLFPACK_ACTIONS)),
            'wolf_2': spaces.Discrete(len(WOLFPACK_ACTIONS)),
            'prey': spaces.Discrete(len(WOLFPACK_ACTIONS)),
        }
        self.state_space = spaces.Box(low=0, high=255, shape=self.state_shape, dtype=np.uint8)

    def seed(self, seed=None):
        self.randomizer, seed = seeding.np_random(seed)
        self.env = Wolfpack(self.randomizer, **self._kwargs)

    def reset(
        self,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> ObsDict:
        if seed is not None:
            self.seed(seed=seed)
        self.env.reset()

        self.agents = self.possible_agents[:]

        if return_info:
            return self.env.observations(), {}
        else:
            return self.env.observations()

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def state(self) -> np.ndarray:
        return ascii_array_to_rgb_array(self.env.grid_world(), self.env.ascii_color_array)

    def action_space(self, agent):
        return self.action_spaces[agent]

    def step(self, actions: ActionDict) -> Tuple[ObsDict, Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, dict]]:

        self.env.step(actions=actions)

        self.agents = []
        for agent in self.env.terminations.keys():
            if (not self.env.terminations[agent]) and (not self.env.truncations[agent]):
                self.agents.append(agent)

        return (
            self.env.observations(),
            self.env.rewards,
            self.env.terminations,
            self.env.truncations,
            self.env.infos,
        )

    def close(self):
        self.env.close()

    def render(self) -> None | np.ndarray | str | List:
        return self.env.render()


def wolfpack_env_creator(config: Dict[str, Any] = {
    'r_lone': 1.0,
    'r_team': 5.0,
    'r_prey': 0.1,
    'max_cycles': 1024,
}) -> WolfpackEnv:
    env = WolfpackEnv(
        r_lone=config['r_lone'],
        r_team=config['r_team'],
        r_prey=config['r_prey'],
        max_cycles=config['max_cycles'],
    )
    return env


wolfpack_env_default_config: Dict[str, Any] = {
    'r_lone': 1.0,
    'r_team': 5.0,
    'r_prey': 0.1,
    'max_cycles': 1024,
}
