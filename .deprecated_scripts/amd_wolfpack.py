import os
import sys

import gymnasium as gym
import numpy as np
import ray
import torch
from gymnasium import spaces
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
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

    config = AMDConfig().environment(
        env=env_name,
        env_config={
            'r_lone': 1.0,
            'r_team': 5.0,
            'r_prey': 0.0,
            'coop_radius': 4,
            'max_cycles': 1024,
        },
        clip_actions=True,
    ).rollouts(
        num_rollout_workers=4,
        rollout_fragment_length=128,
    ).training(
        model={
            "conv_filters": [  # 16x21x3
                [16, [4, 4], 2],  # 7x9x16
                [32, [4, 4], 2],  # 2x3x32
                [256, [4, 6], 1],
            ],
            # 'fcnet_hiddens': [32, 32],
        },
        cp_model={
            'fcnet_hiddens': [128, 32],
            # 'fcnet_hiddens': [],
            'fcnet_activation': 'relu',
            # 'fcnet_activation': 'linear',
        },
        train_batch_size=1024,
        lr=1e-4,
        agent_pseudo_lr=1e-4,
        central_planner_lr=1e-4,
        entropy_coeff=0.01,
        gamma=0.99,
        coop_agent_list=['wolf_1', 'wolf_2'],
        # planner_reward_max=0.1,
        planner_reward_max=0.0,
        reward_distribution='tanh',
        force_zero_sum=False,
        # param_assumption='neural',
        param_assumption='softmax',
    ).debugging(log_level="ERROR").framework(framework="torch").resources(
        num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        num_cpus_per_worker=3,
    )

    tune.run(
        AMD,
        # name="amd_with_r_planner_max=0.1",
        name="no_central_planner",
        stop={"timesteps_total": 5000000},
        keep_checkpoints_num=3,
        checkpoint_freq=10,
        local_dir="~/ray_experiment_results/" + env_name,
        config=config.to_dict(),
    )
