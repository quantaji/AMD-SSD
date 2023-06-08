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

# from core.algorithms.amd.amd import AMD, AMDConfig
from core.algorithms.amd_ppo import AMDPPO, AMDPPOConfig
from core.algorithms.amd.wrappers import MultiAgentEnvFromPettingZooParallel as P2M
from core.environments.simple_games.matrix_game import matrix_game_env_creator, coop_stats_fn

if __name__ == "__main__":
    ray.init()

    # to use gpu
    os.environ['RLLIB_NUM_GPUS'] = '1'
    env_name = 'fear_greed_matrix_game'

    register_env(
        env_name,
        lambda config: P2M(matrix_game_env_creator(config)),
    )

    config = AMDPPOConfig().environment(
        env=env_name,
        env_config={
            'fear': 1.0,
            'greed': 1.0,
        },
        clip_actions=True,
    ).rollouts(
        num_rollout_workers=4,
        rollout_fragment_length=128,
    ).training(
        model={
            'fcnet_hiddens': [32, 32],
            'fcnet_activation': 'relu',
        },
        cp_model={
            # 'fcnet_hiddens': [32, 32],
            'fcnet_hiddens': [128, 128],
            'fcnet_activation': 'relu',
            # 'fcnet_activation': 'linear',
        },
        entropy_coeff=0.000,
        train_batch_size=1024,
        lr=1e-2,
        agent_pseudo_lr=1e-2,
        central_planner_lr=1e-2,
        planner_reward_cost=0.0,
        gamma=0.99,
        reward_distribution='tanh',
        planner_reward_max=3.0,
        # force_zero_sum=True,
        force_zero_sum=False,
        param_assumption='softmax',
        # param_assumption='neural',
        # neural_awareness_method='grad',
        awareness_batch_size=32,
        agent_cooperativeness_stats_fn=coop_stats_fn,
    ).debugging(log_level="ERROR").framework(framework="torch").resources(
        num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        num_cpus_per_worker=3,
    )

    tune.run(
        AMDPPO,
        # name="amd_with_r_planner_max=0.1",
        # name="PD_no_planner",
        # name="Test",
        name="AMDPPO_PD_r_max=3.0_neural_approximation",
        stop={"timesteps_total": 500000},
        keep_checkpoints_num=1,
        checkpoint_freq=10,
        local_dir="~/ray_experiment_results/" + env_name,
        config=config.to_dict(),
    )
