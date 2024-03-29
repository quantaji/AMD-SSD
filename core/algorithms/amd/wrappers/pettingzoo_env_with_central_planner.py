from typing import Dict, List, Tuple

import numpy as np
from gymnasium import spaces
from gymnasium.vector.utils import create_empty_array
from pettingzoo.utils.env import ActionDict, AgentID, ObsDict, ParallelEnv

from ..constants import CENTRAL_PLANNER


class ParallelEnvWithCentralPlanner(ParallelEnv):
    """This object add a central planner to the environment"""

    CENTRAL_PLANNER: str = CENTRAL_PLANNER  # agent id of central planner

    possible_individual_agents: List[str]

    # this is used for algorithm to organize infos into other stuff
    central_planner_info_space: spaces.Space

    def __init__(self, env: ParallelEnv):
        super().__init__()
        self.env = env
        env.reset()

        # try if the env have .state() function, if so, the central planner takes the sate, otherwise it returns a flattened dict
        self.have_state = hasattr(self.env, "state_space")
        if self.have_state:
            try:
                self.env.state()
            except:
                self.have_state = False

        # copy variables from original env
        self.metadata = self.env.metadata.copy()
        self.agents = self.env.agents.copy()
        self.possible_agents = self.env.possible_agents.copy()
        self.observation_spaces = self.env.observation_spaces.copy()
        self.action_spaces = self.env.action_spaces.copy()

        # register central planner
        self.possible_individual_agents = self.possible_agents.copy()
        self.possible_agents.append(self.CENTRAL_PLANNER)
        self.agents.append(self.CENTRAL_PLANNER)

        # # add observation space
        if self.have_state:
            self.observation_spaces[self.CENTRAL_PLANNER] = self.env.state_space
        else:
            self.central_planner_obseravation_space_unflattened = spaces.Dict(self.env.observation_spaces)
            self.observation_spaces[self.CENTRAL_PLANNER] = spaces.flatten_space(self.central_planner_obseravation_space_unflattened)

        # # add action space
        action_space = {}
        for agent_id in self.possible_individual_agents:
            action_space[agent_id] = spaces.Box(low=-1.0, high=1.0, shape=(1, ))
        self.central_planner_action_space_unflattened = spaces.Dict(action_space)
        self.action_spaces[self.CENTRAL_PLANNER] = spaces.flatten_space(self.central_planner_action_space_unflattened)

    def observe_central_planner(self, obs: ObsDict):
        if self.have_state:
            return self.env.state()
        else:
            obs_cp = {}
            for agent_id in self.possible_individual_agents:
                if agent_id in obs.keys():
                    obs_cp[agent_id] = obs[agent_id]
                else:
                    # note for terminated agents, central planner sees zeros
                    obs_cp[agent_id] = create_empty_array(self.observation_space(agent_id))
            return spaces.flatten(self.central_planner_obseravation_space_unflattened, obs_cp)

    def reset(self, seed: int | None = None, return_info: bool = False, options: dict | None = None) -> ObsDict:
        if return_info:
            obs, info = self.env.reset(seed, return_info, options)
        else:
            obs = self.env.reset(seed, return_info, options)

        # now get the observation of central planner
        obs_cp = self.observe_central_planner(obs)
        obs[self.CENTRAL_PLANNER] = obs_cp

        # update living agents
        self.agents = self.env.agents.copy()
        self.agents.append(self.CENTRAL_PLANNER)

        if return_info:
            return obs, info
        else:
            return obs

    def step(self, actions: ActionDict) -> Tuple[ObsDict, Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, dict]]:
        """
        This function collect actions of individual agent, and rewards from central planner. It also
        1. add r_planner to each agent's info
        2. add r_planner as a dict and termination/availability as a dict to central planner's info
        """
        obs, reward, term, trunc, infos = self.env.step(actions=actions)

        obs_cp = self.observe_central_planner(obs)
        obs[self.CENTRAL_PLANNER] = obs_cp

        reward[self.CENTRAL_PLANNER] = 0.0  # central planner does not need reward

        # get the termination and truncation of each agent
        term[self.CENTRAL_PLANNER] = all(term.values())
        trunc[self.CENTRAL_PLANNER] = all(trunc.values())

        # update living agents
        self.agents = self.env.agents.copy()
        if (not term[self.CENTRAL_PLANNER]) and (not trunc[self.CENTRAL_PLANNER]):
            self.agents.append(self.CENTRAL_PLANNER)

        return obs, reward, term, trunc, infos

    def render(self):
        return self.env.render

    def close(self):
        return self.env.close()

    def state(self) -> np.ndarray:
        return self.env.state()

    def observation_space(self, agent: AgentID) -> spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: AgentID) -> spaces.Space:
        return self.action_spaces[agent]

    def unwrapped(self) -> ParallelEnv:
        return self.env
