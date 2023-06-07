from builtins import breakpoint
from dataclasses import replace
from importlib.metadata import metadata
from typing import (Any, Dict, Iterable, Iterator, List, Optional, Tuple, TypeVar)

import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.utils import seeding
from pettingzoo.utils.env import (ActionDict, ActionType, AgentID, ObsDict, ObsType, ParallelEnv)

from core.environments.utils import (ascii_array_to_rgb_array, ascii_dict_to_color_array, ascii_list_to_array)

from ..base.gridworld import GridWorldBase
from .agent import GatheringAgent, GatheringApple
from .constants import (
    GATHERING_ACTIONS,
    GATHERING_AGENT_MAP,
    GATHERING_AGENT_VIEW_TUNE,
    GATHERING_APPLE_NO_ENTRY_STATE,
    GATHERING_BEAM_PLAYER_STATE,
    GATHERING_COLOR,
    GATHERING_MAP,
    GATHERING_MAP_SIZE,
    GATHERING_NO_ENTRY_STATE,
    GATHERING_OBSERVATION_SHAPE,
    GATHERING_ORIENTATION_BOUNDING_BOX,
    GATHERING_ORIENTATION_CHANGE,
    GATHERING_ORIENTATION_TUNE,
    GATHERING_RESOLUTION,
)


class Gathering(GridWorldBase):

    base_world = ascii_list_to_array(GATHERING_MAP)
    world_shape = base_world.shape
    ## get_beam_area, find/place/clear/remove/update_front
    ## place_apple, add_beam_area/empty/place/clear/is; generate pos
    ascii_color_dict = GATHERING_COLOR
    ascii_color_array = ascii_dict_to_color_array(ascii_color_dict)

    resolution = GATHERING_RESOLUTION
    block_size_w, block_size_h = 32, 32
    line_width = 0
    ## Can try to add more players
    agents = ['blue_p', 'red_p']
    # apple_number = GATHERING_APPLE_NUMBER

    def _baseworld_with_apple(self, base_world):
        ## Call it every time when using base world
        ## pos_choice: positions of apples must be set!
        for apple_id in self.apple_dict.keys():
            apple = self.apple_dict[apple_id]
            if apple.is_eaten:
                continue
            posi = apple.position
            base_world[posi[0], posi[1]] = GATHERING_AGENT_MAP['apple']
        return base_world

    def __init__(
        self,
        randomizer: np.random.Generator,
        max_cycles: int = 1024,
        apple_respawn: int = 1,
        apple_number: int = 3,
        player_blood: int = 1,
        tagged_time_number: int = 5,
        r_starv: float = -0.01,
    ):
        self.agent_dict = {}
        self.apple_dict = {}

        self.apple_number = apple_number
        self.tagged_time_number = tagged_time_number
        self.player_blood = player_blood
        self.r_starv = r_starv

        for agent in self.agents:
            self.agent_dict[agent] = GatheringAgent(agent, tagged_time_number=self.tagged_time_number, player_blood=self.player_blood)

        ## in config dict: time for apple respawn
        self.apple_respawn = apple_respawn
        for i in range(self.apple_number):
            ## All apple has the name prefix 'apple_'
            apple_id = 'apple_' + str(i)
            self.apple_dict[apple_id] = GatheringApple(apple_id, self.apple_respawn)

        self.max_cycles = max_cycles

        self.randomizer = randomizer

        super().__init__()

    def reset(self):
        ## reset agents
        x, y = np.where(~np.isin(self.base_world, GATHERING_NO_ENTRY_STATE))

        pos_choice = self.randomizer.choice(len(x), size=(len(self.agents), ), replace=False)
        ori_choice = self.randomizer.choice(4, size=(len(self.agents), ), replace=True)

        for i, agent in enumerate(self.agents):
            self.agent_dict[agent].position = [x[pos_choice[i]], y[pos_choice[i]]]
            self.agent_dict[agent].orientation = ori_choice[i]
            ## reset blood, tagged time, beam etc.
            self.agent_dict[agent]._reset()

        ## reset apples
        x_a, y_a = np.where(~np.isin(self.base_world, GATHERING_APPLE_NO_ENTRY_STATE))
        ## Allow overlapping
        pos_choice_a = self.randomizer.choice(len(y_a), size=(self.apple_number, ), replace=True)
        # ori_choice_a = self.randomizer.choice(4, size=(self.apple_number, ), replace=True)
        for i in range(self.apple_number):
            apple_id = 'apple_' + str(i)
            self.apple_dict[apple_id].position = [x_a[pos_choice_a[i]], y_a[pos_choice_a[i]]]
            #self.apple_dict[apple_id].orientation = ori_choice_a[i]
            ## reset blood, tagged time, beam etc.
            self.apple_dict[apple_id]._reset()

        ## reset counters
        self.reinit()

        self.num_frames = 0

        # update drawing
        self.draw()

    def grid_world(self) -> np.ndarray:
        # load base world
        grid_world = self.base_world.copy()

        # First draw apples
        grid_world = self._baseworld_with_apple(grid_world)
        # add agent position
        for agent_id in self.agent_dict.keys():
            agent = self.agent_dict[agent_id]
            if agent.check_tagged():
                continue
            posi = agent.position
            if grid_world[posi[0], posi[1]] == ' ':
                grid_world[posi[0], posi[1]] = GATHERING_AGENT_MAP[agent_id]
            elif grid_world[posi[0], posi[1]] == 'P':
                grid_world[posi[0], posi[1]] = 'C'
            else:
                grid_world[posi[0], posi[1]] = 'W'

        # add agent orientation
        for agent_id in self.agent_dict.keys():
            agent = self.agent_dict[agent_id]
            if agent.check_tagged():
                continue
            ori_posi = agent.orientation_position
            if (0 <= ori_posi[0]) and (ori_posi[0] < self.world_shape[0]) and (0 <= ori_posi[1]) and (ori_posi[1] < self.world_shape[1]) and grid_world[ori_posi[0], ori_posi[1]] in GATHERING_ORIENTATION_TUNE.keys():
                grid_world[ori_posi[0], ori_posi[1]] = GATHERING_ORIENTATION_TUNE[grid_world[ori_posi[0], ori_posi[1]]]
        '''
        for apple_id in self.apple_dict.keys():
            apple = self.apple_dict[apple_id]
            if apple.is_eaten:
                continue
            posi = apple.position
            if grid_world[posi[0], posi[1]] == ' ':
                grid_world[posi[0], posi[1]] = GATHERING_AGENT_MAP['apple']
            elif grid_world[posi[0], posi[1]] in GATHERING_BEAM_PLAYER_STATE:
                ## 'C' means overlapping of apple and player
                grid_world[posi[0], posi[1]] = 'C'
            else: ## contains overlapping of apple and beam
                grid_world[posi[0], posi[1]] = 'W'
            #if (0 <= ori_posi[0]) and (ori_posi[0] < self.world_shape[0]) and (0 <= ori_posi[1]) and (ori_posi[1] < self.world_shape[1]) and grid_world[ori_posi[0], ori_posi[1]] in GATHERING_ORIENTATION_TUNE.keys():
                #grid_world[ori_posi[0], ori_posi[1]] = GATHERING_ORIENTATION_TUNE[grid_world[ori_posi[0], ori_posi[1]]]
        '''
        # add beam area
        for agent_id in self.agent_dict.keys():
            agent = self.agent_dict[agent_id]
            if agent.check_tagged():
                continue
            if agent.using_beam:
                posi, orii = agent.current_front()
                next_pos = posi + GATHERING_ORIENTATION_CHANGE[orii]
                not_stop = (grid_world[next_pos[0], next_pos[1]] not in GATHERING_NO_ENTRY_STATE)
                while not_stop:
                    if grid_world[next_pos[0], next_pos[1]] == ' ':
                        grid_world[next_pos[0], next_pos[1]] = 'B'
                    elif grid_world[next_pos[0], next_pos[1]] == 'P':
                        grid_world[next_pos[0], next_pos[1]] = 'W'
                    else:  #one question: for C, now set it to not collected
                        grid_world[next_pos[0], next_pos[1]] = 'F'
                    next_pos = next_pos + GATHERING_ORIENTATION_CHANGE[orii]
                    not_stop = (grid_world[next_pos[0], next_pos[1]] not in GATHERING_NO_ENTRY_STATE)

        return grid_world

    def observe(self, agent: AgentID) -> spaces.Space:
        # load base world
        grid_world = self.base_world.copy()
        grid_world = self._baseworld_with_apple(grid_world)
        ## TODO: add other apples? Or they are on the world
        # add agent
        for other_agent_key, other_agent in self.agent_dict.items():
            posi = other_agent.position
            if grid_world[posi[0], posi[1]] == ' ':
                grid_world[posi[0], posi[1]] = GATHERING_AGENT_VIEW_TUNE[agent][GATHERING_AGENT_MAP[other_agent_key]]
            elif grid_world[posi[0], posi[1]] in ['P']:
                grid_world[posi[0], posi[1]] = GATHERING_AGENT_VIEW_TUNE[agent]['P']
            else:
                grid_world[posi[0], posi[1]] = 'W'

        # add orientation
        for other_agent_key, other_agent in self.agent_dict.items():
            ori_posi = other_agent.orientation_position
            if (0 <= ori_posi[0]) and (ori_posi[0] < self.world_shape[0]) and (0 <= ori_posi[1]) and (ori_posi[1] < self.world_shape[1]) and grid_world[ori_posi[0], ori_posi[1]] in GATHERING_ORIENTATION_TUNE.keys():
                grid_world[ori_posi[0], ori_posi[1]] = GATHERING_ORIENTATION_TUNE[grid_world[ori_posi[0], ori_posi[1]]]

        offset = 20
        grid_world = np.pad(grid_world, pad_width=((offset, offset), (offset, offset)), mode='constant', constant_values='0')

        posi = self.agent_dict[agent].position
        orii = self.agent_dict[agent].orientation

        # x_min, x_max, y_min, y_max = np.array() + offset
        x_min = offset + GATHERING_ORIENTATION_BOUNDING_BOX[orii][0] + posi[0]
        x_max = offset + GATHERING_ORIENTATION_BOUNDING_BOX[orii][1] + posi[0]
        y_min = offset + GATHERING_ORIENTATION_BOUNDING_BOX[orii][2] + posi[1]
        y_max = offset + GATHERING_ORIENTATION_BOUNDING_BOX[orii][3] + posi[1]

        observation = grid_world[x_min:x_max + 1, y_min:y_max + 1]
        observation = np.rot90(observation, k=orii)

        return ascii_array_to_rgb_array(observation, self.ascii_color_array)

    def step(self, actions: ActionDict):
        self.reinit()  # reset all rewards etc

        # All agents act
        ## Get newest gridworld with beam area
        grid_world = self.grid_world()
        for agent_key, agent in self.agent_dict.items():
            agent.act(action=actions[agent_key], grid_world=grid_world)

        ## Clear & respawn apple
        for apple_id, apple in self.apple_dict.items():
            apple_pos = apple.position
            if grid_world[apple_pos[0], apple_pos[1]] == 'C':
                apple.get_collected(self.num_frames)
                ## can log collected time
                ## Here record the reward

                apple.eaten_time += 1
            if apple.is_eaten:
                x_a, y_a = np.where(~np.isin(grid_world, GATHERING_APPLE_NO_ENTRY_STATE))
                ## not use randomizer, to ensure not always respawn in the same place
                pos_choice_a = np.random.choice(len(y_a), size=(1, ), replace=True)

                apple.respawn(position=[x_a[pos_choice_a[0]], y_a[pos_choice_a[0]]], current_time_frame=self.num_frames)

        self.num_frames += 1

        # test termination and calc reward
        # MOD: reward is just reward at this time
        for agent_key, agent in self.agent_dict.items():
            posi = agent.position
            if grid_world[posi[0], posi[1]] == 'C':
                #self.rewards[agent_key] = agent.apple_eaten
                self.rewards[agent_key] = 1
            else:
                self.rewards[agent_key] = self.r_starv
        if self.num_frames >= self.max_cycles:
            self.truncations = dict(zip(self.agents, [True] * len(self.agents)))

        if self.render_on:
            pygame.event.pump()
        self.draw()


class GatheringEnv(ParallelEnv):
    metadata = {
        "name": "gathering",
        "render_modes": ["human", "rgb_array", "ansi"],
    }

    observation_shape = (GATHERING_OBSERVATION_SHAPE[0], GATHERING_OBSERVATION_SHAPE[1], 3)
    state_shape = (GATHERING_MAP_SIZE[0], GATHERING_MAP_SIZE[1], 3)

    def __init__(self, **kwargs) -> None:

        self._kwargs = kwargs

        self.env: Gathering

        self.seed()

        self.render_mode = self.env.render_mode
        self.possible_agents = self.env.agents[:]
        self.agents = self.env.agents[:]

        # spaces
        self.observation_spaces = {}
        for agent_key in self.possible_agents:
            self.observation_spaces[agent_key] = spaces.Box(low=0, high=255, shape=self.observation_shape, dtype=np.uint8)

        self.action_spaces = {}
        for agent_key in self.possible_agents:
            self.action_spaces[agent_key] = spaces.Discrete(len(GATHERING_ACTIONS))

        self.state_space = spaces.Box(low=0, high=255, shape=self.state_shape, dtype=np.uint8)

        self.convert_to_rllib_env: bool = False

    def seed(self, seed=None):
        self.randomizer, seed = seeding.np_random(seed)
        self.env = Gathering(self.randomizer, **self._kwargs)

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

        if self.convert_to_rllib_env:
            return self.env.observations(), {}  # in format of observation and info
        else:
            return self.env.observations()

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def state(self) -> np.ndarray:
        return ascii_array_to_rgb_array(self.env.grid_world, self.env.ascii_color_array)

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


def gathering_env_creator(config: Dict[str, Any] = {
    'max_cycles': 1240,
    'apple_respawn': 3,
    'apple_number': 3,
    'player_blood': 1,
    'tagged_time_number': 5,
    'r_starv': -0.01,
}) -> GatheringEnv:
    env = GatheringEnv(
        max_cycles=config['max_cycles'],
        apple_respawn=config['apple_respawn'],
        apple_number=config['apple_number'],
        player_blood=config['player_blood'],
        tagged_time_number=config['tagged_time_number'],
        r_starv=config['r_starv'],
    )
    env.convert_to_rllib_env = True
    return env


gathering_env_default_config: Dict[str, Any] = {
    'max_cycles': 1024,
    'apple_respawn': 3,
    'apple_number': 3,
    'player_blood': 1,
    'tagged_time_number': 5,
    'r_starv': -0.01,
}
