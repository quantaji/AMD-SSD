import os
import sys
import time

import gymnasium as gym
import numpy as np
import ray
import torch
from gymnasium import spaces
from ray import tune
from ray.rllib.algorithms.dqn import DQN, DQNConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.exploration.epsilon_greedy import EpsilonGreedy
from ray.tune.registry import register_env
from torch import nn

module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)

from core.algorithms.amd.amd import AMD, AMDConfig
from core.algorithms.amd.wrappers import MultiAgentEnvFromPettingZooParallel as P2M
from core.environments.gathering.env import gathring_env_creator


class SimpleMLPModelV2(TorchModelV2, nn.Module):

    def __init__(self, obs_space: gym.Space, act_space: gym.Space, num_outputs, *args, **kwargs):
        TorchModelV2.__init__(self, obs_space, act_space, num_outputs, *args, **kwargs)
        nn.Module.__init__(self)

        self.flattened_obs_space = spaces.flatten_space(obs_space)
        self.obs_space = obs_space
        self.action_space = act_space

        self.model = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(self.flattened_obs_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
        )

        self.policy_fn = nn.Linear(32, num_outputs)
        self.value_fn = nn.Linear(32, 1)

    def forward(self, input_dict, state, seq_lens):
        model_out = self.model(input_dict["obs"].to(torch.float32) / 255)
        self._value_out = self.value_fn(model_out)
        return self.policy_fn(model_out), state

    def value_function(self):
        return self._value_out.flatten()


if __name__ == "__main__":
    ray.init()

    # to use gpu
    os.environ['RLLIB_NUM_GPUS'] = '1'
    env_name = 'gathering'

    register_env(
        env_name,
        lambda config: P2M(gathring_env_creator(config)),
    )
    ModelCatalog.register_custom_model("SimpleMLPModelV2", SimpleMLPModelV2)

    config = DQNConfig().multi_agent(
        policies=['predator'],
        policy_mapping_fn=(lambda agent_id, *args, **kwargs: {
            'blue_p': 'predator',
            'red_p': 'predator',
        }[agent_id]),
    ).environment(
        env=env_name,
        env_config={
            'apple_respawn': 10,
            'max_cycles': 1000,
        },
        clip_actions=True,
    ).rollouts(
        num_rollout_workers=4,
        rollout_fragment_length=128,
    ).training(
        model={
            "custom_model": "SimpleMLPModelV2",
            "custom_model_config": {},
        },
        train_batch_size=1000,
        lr=1e-4,
        gamma=0.99,
        v_min=0.0,
        v_max=10.0,
        # double_q=True,
    ).debugging(log_level="ERROR").framework(framework="torch").resources(
        num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        num_cpus_per_worker=3,
    )
    explore_config = {
        "type": EpsilonGreedy,
        "initial_epsilon": 1.0,
        "final_epsilon": 0.1,
        "epsilon_timesteps": 100000,
    }
    config.explore = True,
    config.exploration_config = explore_config

    # print(config.exploration_config)

    algo = config.build().from_checkpoint('/home/quanta/gather_res/DQN_gathering_06939_00000_0_2023-05-21_00-59-51/checkpoint_000970')

    worker = algo.workers.local_worker()
    policy_map = worker.policy_map
    policy_mapping_fn = (lambda agent_id, *args, **kwargs: {
        'blue_p': 'predator',
        'red_p': 'predator',
    }[agent_id])
    for policy_id in policy_map.keys():
        print(policy_id)
    env = P2M(gathring_env_creator({
        'apple_respawn': 10,
        'max_cycles': 1000,
    }))
    obs, _ = env.reset()
    terminated = truncated = False

    timestep = 0

    env.par_env.env.render_mode = 'human'  # 'human'

    env.render()
    render_result = []

    while not terminated and not truncated:

        action = {}
        for agent_id in obs.keys():
            action[agent_id] = algo.compute_single_action(
                observation=obs[agent_id],
                policy_id=policy_mapping_fn(agent_id),
                # explore=False,
            )
        # action = algo.compute_single_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        terminated = terminated['__all__']
        truncated = truncated['__all__']
        timestep += 1

        if env.par_env.env.render_mode == 'human':
            time.sleep(0.2)
            env.render()
        else:
            render_img = env.render()
            render_result.append(render_img)
