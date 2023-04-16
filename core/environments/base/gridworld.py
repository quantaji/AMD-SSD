import numpy as np
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
from gymnasium.utils import seeding
from core.environments.base.agent import GridWorldAgentBase
from core.environments.utils import ascii_array_to_rgb_array, ascii_array_to_str
from gymnasium import spaces, logger
import pygame

# the axes look like
# graphic is here to help me get my head in order
# WARNING: increasing array position in the direction of down
# so for example if you move_left when facing left
# your y position decreases.
# -       ^       +
#         |
#         U
#         P
# <--LEFT*RIGHT---->
#         D
#         O
#         W
#         N
# +       |
# this follows the rules of PIL and openCV, where (0, 0) stands for top-left
# [x, _] is up-to-down axis, [_, y] is left-to-right axis

# I choose to use pettingzoo for following reasons:
# (1) pettingzoo.test.api_test can help test the env
# (2) rllib can convert pettingzoo env to rllib env


class GridWorldBase:

    # a grid world environment have four different descriptions of the world

    # (1) base_world: the fixed position, like walls, this is fixed at the beginning of the game or through the game. This is referred as base_world in the code, it is in ascii form, meaning that is is an np.ndarray of string/char
    # (2) grid_world: this include the agents (players, prey, predator, etc). This is also in ascii format, referred as grid_world in the code, and also add some effect of color, for example, lighten of color in orientation
    # (3) agent_observation: add some other color effects, or croped observation, like other agent and myself.

    # I think the observation of agent is also rgb image, so it can use grid_img or grid_world as well

    # Now I know there are a few major block for writing an environment
    # (1) rules and updates
    # (2) how to observe, render to figure

    # world related
    base_world: np.ndarray
    world_shape: Tuple[int, int]
    ascii_color_dict: Dict[str, Tuple[int, int, int]]  # an ascii string, e.g. ' ', '@', denoting a grid's state, map to a RGB color [0-255, 0-255, 0-255]

    # agent and action related
    agents: List[AgentID]
    agent_dict: Dict[AgentID, GridWorldAgentBase]  # a dictionary that gives an agent object, the agent object is then used to update states

    # render related
    resolution: Tuple[int, int]  # (width, hight), e.g.  (960, 720) # (640, 480) # (100, 200)
    render_on: bool = False  # this stores whether we render
    render_mode: Optional[str] = None  # in 'human', "rgb_array", "ansi"
    screen: pygame.Surface  # what we put on screen, frequently updating
    block_size_h: int
    block_size_w: int
    line_width: int = 1
    rect_grid: list[list[pygame.Rect]]  # a 2d list for drawing, axis 0 for high direction,

    # update related
    rewards: Dict[AgentID, float]
    terminations: Dict[AgentID, bool]
    truncations: Dict[AgentID, bool]
    infos: Dict[AgentID, dict]

    num_frames: int

    def __init__(self):
        """Initialization function.
        """
        raise NotImplementedError

    def reinit(self):
        self.rewards = dict(zip(self.agents, [0.0] * len(self.agents)))
        self.terminations = dict(zip(self.agents, [False] * len(self.agents)))
        self.truncations = dict(zip(self.agents, [False] * len(self.agents)))
        self.infos = dict(zip(self.agents, [{}] * len(self.agents)))

    def seed(self, seed=None):
        self.randomizer, _ = seeding.np_random(seed)

    def reset(self, seed: Optional[int] = None):
        """One have to reset all agents, and then get the new observation for each agent
        """
        raise NotImplementedError
        # here is an example code
        # self.seed(seed)
        # self.reset_agents()
        # self.reset_world()
        # self.reinit()
        # self.draw()

    def render(self) -> None | np.ndarray:
        if self.render_mode is None:
            logger.warn("You are calling render method without specifying any render mode.")
            return None

        if (not self.render_on) and (self.render_mode in ['rgb_array', 'human']):
            self.enable_render()

        view = self.grid_world()
        if self.render_mode == 'ansi':
            return ascii_array_to_str(ascii_arr=view)

        elif self.render_mode == 'rgb_array':
            return np.array(pygame.surfarray.pixels3d(self.screen)).transpose(axes=(1, 0, 2))

        elif self.render_mode == 'human':
            pygame.display.flip()

        else:
            raise TypeError('The current render mode: {} is not supported!'.format(self.render_mode))

    def close(self):
        if self.render_on:
            pygame.event.pump()
            pygame.display.quit()
            self.render_on = False

    def enable_render(self):
        self.screen = pygame.display.set_mode(size=self.resolution)
        self.renderOn = True
        self.rect_grid = [[pygame.Rect(
            j * self.block_size_w,
            i * self.block_size_h,
            self.block_size_w,
            self.block_size_h,
        ) for j in range(self.world_shape[1])] for i in range(self.world_shape[0])]
        self.draw()

    def draw(self):
        """For updating self.screen variable
        """
        view = self.grid_world()
        if self.render_on:
            for i in range(self.world_shape[0]):
                for j in range(self.world_shape[1]):
                    pygame.draw.rect(surface=self.screen, color=self.ascii_color_dict[view[i][j]], rect=self.rect_grid[i][j], width=self.line_width)

    def grid_world(self) -> np.ndarray:
        """This function returns the grid world (state). I set it to be in ascii form. 
        """
        raise NotImplementedError

    def observe(self, agent_id: AgentID) -> spaces.Space:
        """Get a dictionary of observations. Usually this is obtained from the state (grid_world)
        """
        raise NotImplementedError

    def step(self, actions: ActionDict) -> Tuple[ObsDict, Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, dict]]:
        """ (1) update action (2) calculate reward and termination (3) info
        """
        raise NotImplementedError
