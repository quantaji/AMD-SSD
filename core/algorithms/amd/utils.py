import functools

import gymnasium as gym
import numpy as np
import pkg_resources
import ray
import torch
from packaging import version
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.env.env_context import EnvContext
from ray.rllib.env.utils import _gym_env_creator
from ray.rllib.utils.error import ERR_MSG_INVALID_ENV_DESCRIPTOR, EnvError
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.typing import EnvCreator, EnvType
from ray.tune.registry import ENV_CREATOR, _global_registry


def get_env_example(config: AlgorithmConfig) -> EnvType:
    """These code are copied and modified from Algoirhtm._get_env_id_and_creator"""
    env_specifier = config.env
    env_creator: EnvCreator = None
    if isinstance(env_specifier, str):
        # An already registered env.
        if _global_registry.contains(ENV_CREATOR, env_specifier):
            env_creator = _global_registry.get(ENV_CREATOR, env_specifier)

        # A class path specifier.
        elif "." in env_specifier:

            def env_creator_from_classpath(env_context):
                try:
                    env_obj = from_config(env_specifier, env_context)
                except ValueError:
                    raise EnvError(ERR_MSG_INVALID_ENV_DESCRIPTOR.format(env_specifier))
                return env_obj

            env_creator = env_creator_from_classpath
        # Try gym/PyBullet/Vizdoom.
        else:
            env_creator = functools.partial(_gym_env_creator, env_descriptor=env_specifier)

    elif isinstance(env_specifier, type):
        env_id = env_specifier  # .__name__

        if config["remote_worker_envs"]:
            # Check gym version (0.22 or higher?).
            # If > 0.21, can't perform auto-wrapping of the given class as this
            # would lead to a pickle error.
            gym_version = pkg_resources.get_distribution("gym").version
            if version.parse(gym_version) >= version.parse("0.22"):
                raise ValueError("Cannot specify a gym.Env class via `config.env` while setting "
                                 "`config.remote_worker_env=True` AND your gym version is >= "
                                 "0.22! Try installing an older version of gym or set `config."
                                 "remote_worker_env=False`.")

            @ray.remote(num_cpus=1)
            class _wrapper(env_specifier):
                # Add convenience `_get_spaces` and `_is_multi_agent`
                # methods:
                def _get_spaces(self):
                    return self.observation_space, self.action_space

                def _is_multi_agent(self):
                    from ray.rllib.env.multi_agent_env import MultiAgentEnv

                    return isinstance(self, MultiAgentEnv)

            env_creator = lambda cfg: _wrapper.remote(cfg)
        # gym.Env-subclass: Also go through our RLlib gym-creator.
        elif issubclass(env_specifier, gym.Env):
            env_creator = functools.partial(
                _gym_env_creator,
                env_descriptor=env_specifier,
                auto_wrap_old_gym_envs=config.get("auto_wrap_old_gym_envs", True),
            )
        # All other env classes: Call c'tor directly.
        else:
            env_creator = lambda cfg: env_specifier(cfg)

    env_context = EnvContext(config.env_config, worker_index=0)

    env = env_creator(env_context)

    return env


def get_availability_mask(cp_t: np.ndarray, cp_eps_id: np.ndarray, ag_t: np.ndarray, ag_eps_id: np.ndarray):
    """Some times some agent dies but planner does not. So We have to compute a maske of what time step each agent is available. This function assumes that when agent dies, its batch does not have the correspongding time step t.
    """
    dtype = {'names': ['t', 'eps_id'], 'formats': [int, int]}

    cp_comb = np.vstack([cp_eps_id, cp_t]).T.copy().view(dtype).reshape(-1)
    ag_comb = np.vstack([ag_eps_id, ag_t]).T.copy().view(dtype).reshape(-1)

    # central planner have more time instance than agent so we don't need to union
    mask = np.in1d(cp_comb, ag_comb, assume_unique=True)

    return mask


def discounted_cumsum_factor_matrix(
    eps_id: np.ndarray | torch.Tensor,
    t: np.ndarray | torch.Tensor,
    gamma: float = 1.0,
) -> np.ndarray | torch.Tensor:
    """
    Given a batch of length T, return a TxT matrix of following element
    M_ij = gamma^{t_j} if they are of same episode, else 0
    """
    t_diff = t.reshape(1, -1) - 0 * t.reshape(-1, 1)
    return (gamma**t_diff) * (eps_id.reshape(-1, 1) == eps_id.reshape(1, -1))


def action_to_reward(
    actions: np.ndarray | torch.Tensor,
    availability: np.ndarray | torch.Tensor,
    appearance: np.ndarray,
    reward_max: float,
    zero_sum: bool,
) -> np.ndarray | torch.Tensor:
    reward = reward_max * actions
    if zero_sum:
        reward = reward - reward[:, appearance].mean(-1).reshape(-1, 1)
    return reward * availability
