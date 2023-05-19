import numpy as np
from pettingzoo.utils.env import ActionType, AgentID

from core.environments.base.constants import *
from core.environments.gathering.constants import (GATHERING_MAP_SIZE, GATHERING_NO_ENTRY_STATE, GATHERING_PLAYER_BLOOD, GATHERING_TAGGED_TIME)

from ..base.agent import GridWorldAgentBase


class GatheringAgent(GridWorldAgentBase):

    no_entry_grid_state_list = GATHERING_NO_ENTRY_STATE
    map_x, map_y = GATHERING_MAP_SIZE

    def __init__(self, agent_id: AgentID) -> None:
        super().__init__(agent_id)
        #self.orientation(0)
        #self.position(pos=np.random.randint(low=[0,0], high=[GATHERING_MAP_SIZE], size=(2,)))
        self.using_beam = False
        self.is_tagged = False
        self.num_hit_by_beam = False
        self.tagged_time = 0
        self.apple_eaten = 0
        self.is_agent = False  # Agent=blue, other=red; Is needed?

    def _reset(self):
        self.using_beam = False
        self.is_tagged = False
        self.num_hit_by_beam = False
        self.tagged_time = 0
        self.apple_eaten = 0

    def current_front(self):
        return self._position, self._orientation

    def stand_still(self):
        self.using_beam = False
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
        if self.tagged_time == 1:
            self.respawn()
            return
        self.tagged_time -= 1

    def respawn(self):
        self.is_tagged = False
        self.tagged_time = 0

    def act(self, action: ActionType, grid_world: np.ndarray):
        ## First check agent state
        ## If use beam in the last round, use it to convert it back
        self.using_beam = False
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


class GatheringApple():
    no_entry_grid_state_list = GATHERING_NO_ENTRY_STATE
    map_x, map_y = GATHERING_MAP_SIZE

    def __init__(self, agent_id: AgentID, apple_respawn) -> None:
        #super().__init__(agent_id)
        self.agent_id = agent_id
        self.position = None
        #self.orientation = None
        self.apple_respawn = apple_respawn
        #self.orientation(0)
        #self.position(pos=np.random.randint(low=[0,0], high=[GATHERING_MAP_SIZE], size=(2,)))
        self.is_eaten = False
        self.collected_time = 0.
        ## Just counter for apple to record the number get eaten
        self.eaten_time = 0
        self.respawn_time_frame = 0

    def respawn(self, position, current_time_frame):
        #self.orientation(0)
        if self.respawn_time_frame == current_time_frame:
            self.position = position
            self.is_eaten = False
            self.collected_time = 0.
            self.respawn_time_frame = 0.

    def _reset(self):
        self.is_eaten = False
        self.collected_time = 0.
        self.respawn_time_frame = 0

    def get_collected(self, time):
        self.is_eaten = True
        self.collected_time = time
        self.respawn_time_frame = time + self.apple_respawn
