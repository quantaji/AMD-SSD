from typing import Optional

from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv
from ray.rllib.env.multi_agent_env import ENV_STATE, MultiAgentEnv

STATE_SPACE = 'state_space'


class MultiAgentEnvFromPettingZooParallel(MultiAgentEnv):

    def __init__(self, env: ParallelEnv):

        self.par_env = env
        self.par_env.reset()

        self.observation_space = spaces.Dict(self.par_env.observation_spaces)
        self.action_space = spaces.Dict(self.par_env.action_spaces)

        self._agent_ids = set(self.par_env.possible_agents)

        # see if state is callable
        try:
            getattr(self.par_env, ENV_STATE)()
            self.state = getattr(self.par_env, ENV_STATE)
        except:
            pass

        # see if it specifies state space
        if hasattr(self.par_env, STATE_SPACE) and isinstance(getattr(self.par_env, STATE_SPACE), spaces.Space):
            self.state_space = getattr(self.par_env, STATE_SPACE)

        self._obs_space_in_preferred_format = self._check_if_obs_space_maps_agent_id_to_sub_space()
        self._action_space_in_preferred_format = self._check_if_action_space_maps_agent_id_to_sub_space()

        super().__init__()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = self.par_env.reset(seed=seed, options=options)
        return obs, info

    def step(self, action_dict):
        obss, rews, terminateds, truncateds, infos = self.par_env.step(action_dict)
        terminateds["__all__"] = all(terminateds.values())
        truncateds["__all__"] = all(truncateds.values())
        return obss, rews, terminateds, truncateds, infos

    def close(self):
        self.par_env.close()

    def render(self):
        return self.par_env.render()

    @property
    def get_sub_environments(self):
        return self.par_env.unwrapped
