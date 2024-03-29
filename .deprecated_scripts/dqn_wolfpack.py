import os
import sys

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
from core.environments.wolfpack import wolfpack_env_creator

if __name__ == "__main__":
    ray.init()

    # to use gpu
    os.environ['RLLIB_NUM_GPUS'] = '1'
    env_name = 'wolfpack'

    register_env(
        env_name,
        lambda config: P2M(wolfpack_env_creator(config)),
    )

    config = DQNConfig().multi_agent(
        policies=['predator', 'prey'],
        policy_mapping_fn=(lambda agent_id, *args, **kwargs: {
            'wolf_1': 'predator',
            'wolf_2': 'predator',
            'prey': 'prey',
        }[agent_id]),
    ).environment(
        env=env_name,
        env_config={
            'r_lone': 1.0,
            'r_team': 5.0,
            'r_prey': 0.0,  # for an average of 50 steps, cooperation will lead to positive reward while defective will not
            'r_starv': -0.01,
            'coop_radius': 4,
            'max_cycles': 1024,
        },
        clip_actions=True,
    ).rollouts(
        num_rollout_workers=4,
        rollout_fragment_length=128,
    ).training(
        # model={
        #     "conv_filters": [  # 16x21x3
        #         [16, [4, 4], 2],  # 7x9x16
        #         [32, [4, 4], 2],  # 2x3x32
        #         [256, [4, 6], 1],
        #     ],
        # },
        model={
            "conv_filters": [
                [16, [4, 4], 2],
                [32, [4, 4], 2],
                [128, [4, 6], 1],
            ],
            # "post_fcnet_hiddens": [32, 32],
        },
        train_batch_size=1024,
        lr=1e-4,
        gamma=0.99,
        v_min=0.0,
        v_max=10.0,
        double_q=True,
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

    print(config.exploration_config)

    tune.run(
        DQN,
        name='dqn-with-living-panelty-dual',
        stop={"timesteps_total": 1000000},
        keep_checkpoints_num=10,
        checkpoint_freq=1,
        local_dir="~/ray_experiment_results/" + env_name,
        config=config.to_dict(),
    )
