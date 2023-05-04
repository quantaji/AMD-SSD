from ..base.agent import GridWorldAgentBase
from pettingzoo.utils.env import (
    ActionType,
    AgentID,
)
from core.environments.gathering.constants import GATHERING_NO_ENTRY_STATE, GATHERING_MAP_SIZE, GATHERING_PLAYER_BLOOD, GATHERING_TAGGED_TIME
from core.environments.base.constants import *

import numpy as np

class GatheringAgent(GridWorldAgentBase):

    no_entry_grid_state_list = GATHERING_NO_ENTRY_STATE
    map_x, map_y = GATHERING_MAP_SIZE

    def __init__(self, agent_id: AgentID) -> None:
        super().__init__(agent_id)
        self.orientation(0)
        self.position(pos=np.random.randint(low=[0,0], high=[GATHERING_MAP_SIZE], size=(2,)))
        self.using_beam = False
        self.is_tagged = False
        self.num_hit_by_beam = False
        self.tagged_time = 0
        self.apple_eaten = 0
        self.is_agent = False # Agent=blue, other=red; Is needed?
        
    def _reset(self):
        self.using_beam = False
        self.is_tagged = False
        self.num_hit_by_beam = False
        self.tagged_time = 0
        self.apple_eaten = 0

    @property
    def current_front(self):
        return self._position, self._orientation

    def stand_still(self):
        self.using_beam=False
        pass 
    # perhaps no need
    def check_tagged(self):
        if self.is_tagged:
            self.using_beam = False
            return True
        else:
            return False

    def use_beam(self):
        self.using_beam = True
    
    def get_hit(self):
        self.num_hit_by_beam += 1
        if self.num_hit_by_beam >= GATHERING_PLAYER_BLOOD:
            self.num_hit_by_beam = 0
            self.using_beam = False
            self.is_tagged = True
            self.tagged_time = GATHERING_TAGGED_TIME
    
    def recover(self):
        self.tagged_time -= 1
        if self.tagged_time == 0:
            self.respawn()
            return

    
    def respawn(self):
        self.is_tagged = False
        self.tagged_time = 0

    def act(self, action: ActionType, grid_world: np.ndarray):
        ## First check agent state
        if self.check_tagged():
            ## player will come back to the original place in next round
            self.recover()
        if action == 7:
            self.use_beam()
            ## need to clear grid?
        if grid_world[self.position[0], self.position[1]] == 'F':
            self.get_hit()
        ## collect apple
        if grid_world[self.position[0], self.position[1]] == 'C':
            self.apple_eaten += 1
        else:
            self.move(action=action, grid_world=grid_world)

class GatheringApple(GridWorldAgentBase):
    no_entry_grid_state_list = GATHERING_NO_ENTRY_STATE
    map_x, map_y = GATHERING_MAP_SIZE

    def __init__(self, agent_id: AgentID) -> None:
        super().__init__(agent_id)
        self.orientation(0)
        self.position(pos=np.random.randint(low=[0,0], high=[GATHERING_MAP_SIZE], size=(2,)))
        self.is_eaten = False
        self.collected_time = 0.

    def respawn(self, position):
        self.orientation(0)
        self.position(pos=np.random.randint(low=[0,0], high=[GATHERING_MAP_SIZE], size=(2,)))
        self.is_eaten = False
        self.collected_time = 0.

    def _reset(self):
        self.is_eaten = False
        self.collected_time = 0.
    
    def get_collected(self, time):
        self.is_eaten = True
        self.collected_time = time