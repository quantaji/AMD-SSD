from copy import deepcopy
from typing import Optional, Tuple

import numpy as np
from gymnasium import spaces
from gymnasium.core import Env
from gymnasium.spaces import Space
from gymnasium.vector.utils import create_empty_array
from ray.rllib.env.multi_agent_env import ENV_STATE, MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict

from ..constants import CENTRAL_PLANNER, STATE_SPACE


class MultiAgentEnvWithCentralPlanner(MultiAgentEnv):

    CENTRAL_PLANNER: str = CENTRAL_PLANNER
    observation_space: Optional[spaces.Dict] = None
    action_space: Optional[spaces.Dict] = None
    _agent_ids: set[str]
    _individual_agent_ids: set[str]

    env_state_space: Space
    agent_action_space: Space
    agent_presence_space: Space

    central_planner_observation_space_unflattened: Space

    def __init__(self, env: MultiAgentEnv):

        assert isinstance(env, MultiAgentEnv), 'The environment must be a MultiAgentEnv'

        self.env: MultiAgentEnv = env
        self.env.reset()

        # the name of central planner is always 'central_planner

        # try if the env have .state() function, if so, the central planner takes the sate, otherwise it returns a flattened dict
        self.have_state = hasattr(self.env, STATE_SPACE) and hasattr(self.env, ENV_STATE) and callable(getattr(self.env, ENV_STATE))
        if self.have_state:
            try:
                getattr(self.env, ENV_STATE)()
                self.state = getattr(self.env, ENV_STATE)
                self.state_space = getattr(self.env, STATE_SPACE)
            except:
                self.have_state = False

        # copy variables from original env
        self._agent_ids = deepcopy(self.env.get_agent_ids())
        self.observation_space = deepcopy(spaces.Dict(self.env.observation_space))
        self.action_space = deepcopy(spaces.Dict(self.env.action_space))

        # register central planner
        self._individual_agent_ids = deepcopy(set(self._agent_ids))
        self._agent_ids.add(self.CENTRAL_PLANNER)

        # add observation space
        # central planner's observation space is concatentation of state x actions x agent presents (binary)
        if self.have_state:
            self.env_state_space = getattr(self.env, STATE_SPACE)
        else:
            self.env_state_space = spaces.Dict(self.env.observation_space)

        self.agent_action_space = self.env.action_space

        agents = self.env.get_agent_ids()
        self.agent_presence_space = spaces.Dict(dict(zip(agents, [spaces.MultiBinary(1)] * len(agents))))

        self.central_planner_observation_space_unflattened = spaces.Dict({
            'state': self.env_state_space,
            'action': self.agent_action_space,
            'presence': self.agent_presence_space,
        })
        self.observation_space[self.CENTRAL_PLANNER] = spaces.flatten_space(self.central_planner_observation_space_unflattened)

        # add action space, the reward by planner
        cp_action_space = {}
        for agent_id in self.env.get_agent_ids():
            cp_action_space[agent_id] = spaces.Box(low=-1.0, high=1.0, shape=(1, ))
        self.central_planner_action_space_unflattened = spaces.Dict(cp_action_space)
        self.action_space[self.CENTRAL_PLANNER] = spaces.flatten_space(self.central_planner_action_space_unflattened)

        self._obs_space_in_preferred_format = self._check_if_obs_space_maps_agent_id_to_sub_space()
        self._action_space_in_preferred_format = self._check_if_action_space_maps_agent_id_to_sub_space()

        super().__init__()

    def observe_central_planner(self, obss: MultiAgentDict, acts: MultiAgentDict = None):
        state, action, presence = {}, {}, {}
        for agent_id in self.env.get_agent_ids():
            if agent_id in obss.keys():
                state[agent_id] = obss[agent_id]
            else:
                state[agent_id] = create_empty_array(self.observation_space[agent_id])  # this might be bugful because shape will be different, but this will be wiped out when flattened

            if acts and agent_id in acts.keys():
                action[agent_id] = acts[agent_id]
                presence[agent_id] = np.array([1], dtype=np.int8)
            else:
                action[agent_id] = create_empty_array(self.agent_action_space[agent_id])
                presence[agent_id] = np.array([0], dtype=np.int8)

        if self.have_state:
            state = self.state()

        cp_obs_uf = {
            'state': state,
            'action': action,
            'presence': presence,
        }

        return spaces.flatten(self.central_planner_observation_space_unflattened, cp_obs_uf)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> Tuple[MultiAgentDict, MultiAgentDict]:
        obss, infos = self.env.reset(seed=seed, options=options)

        # now get the observation of central planner
        obss[self.CENTRAL_PLANNER] = self.observe_central_planner(obss)

        return obss, infos

    def step(self, action_dict: MultiAgentDict) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:

        obss, rews, terminateds, truncateds, infos = self.env.step(action_dict)

        terminateds[self.CENTRAL_PLANNER] = terminateds["__all__"]
        truncateds[self.CENTRAL_PLANNER] = truncateds["__all__"]
        infos[self.CENTRAL_PLANNER] = {}

        rews[self.CENTRAL_PLANNER] = 0.0

        obss[self.CENTRAL_PLANNER] = self.observe_central_planner(obss=obss, acts=action_dict)

        return obss, rews, terminateds, truncateds, infos

    def render(self) -> None:
        return self.env.render()

    def close(self):
        return self.env.close()

    def unwrapped(self) -> Env:
        return self.env
