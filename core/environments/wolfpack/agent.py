import numpy as np
from pettingzoo.utils.env import ActionType, AgentID

from ..base.agent import GridWorldAgentBase
from .constants import WOLFPACK_NO_ENTRY_STATE


class WolfpackAgent(GridWorldAgentBase):

    no_entry_grid_state_list = WOLFPACK_NO_ENTRY_STATE

    def __init__(self, agent_id: AgentID) -> None:
        super().__init__(agent_id)

    def act(self, action: ActionType, grid_world: np.ndarray):
        self.move(action=action, grid_world=grid_world)
