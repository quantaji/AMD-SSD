"""
I searched over internet, there is not existing implementation for gridworld agent's action and update
I learn a lot from https://github.com/vermashresth/sequential_social_dilemma_games/tree/ppo_comm_policy/social_dilemmas/envs and https://github.com/dkkim93/gym-wolfpack/blob/master/gym_env/wolfpack/agent.py
The key principle is that
(1) the agent updates its position, orientation, or other internal states, generally using an act() funciton,
(2) we can put additional process or ever rewrite act function, we can define new function for each custom agent class
(3) given a grid_world (an ndarray denoting the world state), and predifined viewing scope, the agent should be able to return an observation of the world

further we have to also define the world size and observation size for agent
"""
from typing import List, Tuple

import numpy as np
from pettingzoo.utils.env import ActionType, AgentID

from .constants import (ACTION_ORIENTATION_CHANGE, ACTION_POSITION_CHANGE, ORIENTATION_CHANGE)


class GridWorldAgentBase:

    grid_world_shape: Tuple[int, int]
    observation_shape: Tuple[int, int]

    no_entry_grid_state_list: List[str] = ['@', '0']

    _orientation: int
    _position: np.ndarray

    def __init__(
        self,
        agent_id: AgentID,
    ) -> None:

        self._agent_id = agent_id

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, pos: np.ndarray):
        if not isinstance(pos, np.ndarray):
            pos = np.array(pos)
        assert pos.shape == (2, ), "The given position array is not of shape (2,)"
        self._position = pos

    @property
    def orientation(self):
        return self._orientation

    @orientation.setter
    def orientation(self, ori: int):
        self._orientation = ori % 4

    @property
    def orientation_position(self):
        return self.position + ORIENTATION_CHANGE[self.orientation]

    def act(self, action: ActionType, grid_world: np.ndarray):
        """This function is designed for updating the agent's internal state and position/orientation. The self.move function have already implement the position/orientation part.
        Args:
            action: the action_id you choose, 
            grid_world: an np.ndarray specify the state of current location
        """
        raise NotImplementedError

    def move(self, action: ActionType, grid_world: np.ndarray):
        """update position and orientation
        """
        new_pos = self._new_position(action)
        new_ori = self._new_orientation(action)

        # update position
        if grid_world[new_pos[0], new_pos[1]] not in self.no_entry_grid_state_list:
            self._position = new_pos

        # update orientation
        self._orientation = new_ori

    def _new_position(self, action: ActionType) -> np.ndarray:
        new_pos = self.position.copy()
        if action in ACTION_POSITION_CHANGE.keys():
            #new_pos += ACTION_POSITION_CHANGE[action]
            new_pos += ACTION_POSITION_CHANGE[(action + self.orientation) % 4]
        return new_pos

    def _new_orientation(self, action: ActionType) -> int:
        if action in ACTION_ORIENTATION_CHANGE.keys():
            return (self.orientation + ACTION_ORIENTATION_CHANGE[action]) % 4
        else:
            return self.orientation
