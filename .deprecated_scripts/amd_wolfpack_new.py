import os
import sys

import gymnasium as gym
import numpy as np
import ray
import supersuit as ss
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
from core.algorithms.amd.wrappers import \
    MultiAgentEnvFromPettingZooParallel as P2M
from core.environments.wolfpack import wolfpack_env_creator

if __name__ == "__main__":
    ray.init()

    # to use gpu
    os.environ['RLLIB_NUM_GPUS'] = '1'
    env_name = 'wolfpack'

    register_env(env_name, lambda config: P2M(ss.normalize_obs_v0(ss.dtype_v0(
        wolfpack_env_creator(config),
        np.float32,
    ))))

    config = AMDConfig().environment(
        env=env_name,
        env_config={
            'r_lone': 1.0,
            'r_team': 5.0,
            'r_prey': 0.01,
            'r_starv': -0.01,
            'max_cycles': 1024,
        },
        clip_actions=True,
    ).rollouts(
        # num_rollout_workers=6,
        num_rollout_workers=0,
        rollout_fragment_length=1024,
        # num_envs_per_worker=4,
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
        cp_model={
            'fcnet_hiddens': [256, 128],
            'fcnet_activation': 'relu',
            "use_lstm": True,
            "lstm_cell_size": 32,
            "max_seq_len": 32,
        },
        # train_batch_size=4 * 6 * 1024,
        train_batch_size=128,
        lr=1e-4,
        lr_schedule=[[0, 0.00136], [20000000, 0.000028]],
        gamma=0.99,
        lambda_=0.95,
        entropy_coeff=0.000687,
        vf_loss_coeff=0.5,
        agent_pseudo_lr=1e-4,
        central_planner_lr=1e-4,
        coop_agent_list=['wolf_1', 'wolf_2'],
        planner_reward_max=0.01,
        # planner_reward_max=0.0,
        reward_distribution='tanh',
        force_zero_sum=False,
        # param_assumption='neural',
        param_assumption='softmax',
    ).debugging(log_level="ERROR").framework(framework="torch").resources(
        num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        num_cpus_per_worker=1,
    )

    tune.run(
        AMD,
        # name="amd_with_r_planner_max=0.1",
        name="AMD_no_cp",
        stop={"timesteps_total": 60 * (10**6)},
        keep_checkpoints_num=3,
        checkpoint_freq=10,
        local_dir="~/ray_test/" + env_name,
        config=config.to_dict(),
    )
