import os
import sys

import gymnasium as gym
import numpy as np
import ray
import torch
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env
from torch import nn

module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)

from core.environments.wolfpack import wolfpack_env_creator

if __name__ == "__main__":
    ray.init()

    # to use gpu
    os.environ['RLLIB_NUM_GPUS'] = '1'

    env_name = 'wolfpack'
    config_template = {
        'r_lone': 1.0,
        'r_team': 5.0,
        'r_prey': 0.0,
        'max_cycles': 1024,
    }

    register_env(env_name, lambda config: ParallelPettingZooEnv(wolfpack_env_creator(config)))

    config = PPOConfig().environment(
        env=env_name,
        env_config=config_template,
    ).rollouts(num_rollout_workers=4, rollout_fragment_length=128).training(
        model={
            "conv_filters": [  # 16x21x3
                [16, [4, 4], 2],  # 7x9x16
                [32, [4, 4], 2],  # 2x3x32
                [256, [4, 6], 1],
            ],
        },
        train_batch_size=512,
        lr=1e-4,
        gamma=0.99,
        lambda_=0.9,
        use_gae=True,
        clip_param=0.4,
        grad_clip=None,
        entropy_coeff=0.1,
        vf_loss_coeff=0.25,
        sgd_minibatch_size=64,
        num_sgd_iter=10,
    ).debugging(log_level="ERROR", ).framework(framework="torch", ).resources(
        num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        num_cpus_per_worker=3,
    ).multi_agent(
        policies=['predator', 'prey'],
        policy_mapping_fn=(lambda agent_id, *args, **kwargs: {
            'wolf_1': 'predator',
            'wolf_2': 'predator',
            'prey': 'prey',
        }[agent_id]),
    )

    tune.run(
        "PPO",
        name="PPO",
        stop={"timesteps_total": 5000000},
        keep_checkpoints_num=3,
        checkpoint_freq=10,
        local_dir="~/ray_results_new/" + env_name,
        config=config.to_dict(),
    )
