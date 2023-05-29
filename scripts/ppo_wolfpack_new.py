import os
import sys

import gymnasium as gym
import numpy as np
import ray
import torch
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env
from torch import nn
import supersuit as ss

module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)

from core.environments.wolfpack import wolfpack_env_creator
from core.algorithms.amd.wrappers import MultiAgentEnvFromPettingZooParallel as P2M

if __name__ == "__main__":
    ray.init()

    # to use gpu
    os.environ['RLLIB_NUM_GPUS'] = '1'

    env_name = 'wolfpack'
    config_template = {
        'r_lone': 1.0,
        'r_team': 5.0,
        'r_prey': 0.0,
        'r_starv': -0.01,
        'max_cycles': 1024,
    }

    register_env(env_name, lambda config: P2M(ss.normalize_obs_v0(
        ss.dtype_v0(
            wolfpack_env_creator(config),
            np.float32,
        ),
        env_min=-0.5,
        env_max=0.5,
    )))

    config = PPOConfig().environment(
        env=env_name,
        env_config=config_template,
    ).rollouts(
        num_rollout_workers=6,
        rollout_fragment_length=1024,
        num_envs_per_worker=16,
    ).training(
        model={
            "conv_filters": [
                [6, [3, 3], 1],
            ],
            "post_fcnet_hiddens": [32, 32],
            "use_lstm": True,
            "lstm_cell_size": 128,
            "max_seq_len": 32,
        },
        train_batch_size=16 * 6 * 1024,
        lr=1e-4,
        lr_schedule=[[0, 0.00136], [20000000, 0.000028]],
        gamma=0.99,
        entropy_coeff=0.000687,
        vf_loss_coeff=1e-4,
        sgd_minibatch_size=16 * 6 * 1024 // 4,
        num_sgd_iter=10,
    ).debugging(log_level="ERROR", ).framework(framework="torch", ).resources(
        num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        num_cpus_per_worker=1,
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
        name="PPO-config-from-other-paper",
        stop={"timesteps_total": 500 * (10**6)},
        keep_checkpoints_num=3,
        checkpoint_freq=10,
        # local_dir="~/ray_experiment_results/" + env_name,
        local_dir="~/ray_test/" + env_name,
        config=config.to_dict(),
    )
