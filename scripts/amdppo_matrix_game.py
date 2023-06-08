import argparse
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

from core.algorithms.amd.wrappers import MultiAgentEnvFromPettingZooParallel as P2M
from core.algorithms.amd_ppo import AMDPPO, AMDPPOConfig
from core.environments.simple_games.matrix_game import matrix_game_coop_stats_fn, matrix_game_env_creator


def parse_args():
    parser = argparse.ArgumentParser("Adaptive Mechanism Design on Matrix Game")

    parser.add_argument(
        "--game_type",
        type=str,
        default="PrisonerDilemma",
        choices=["PrisonerDilemma", "StagHunt", "Chicken"],
        help='Type of the environment. ["PrisonerDilemma", "StagHunt", "Chicken"]',
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        help="Name of this experiment.",
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        default='~/forl-exp/',
        help="Path to store experiment results. The folder of this experiment is 'exp_dir/env_name/exp_name/'",
    )
    parser.add_argument(
        "--num_rollout_workers",
        type=int,
        default=0,
        help="Number of rollout works. Default is 0, meaning only using local worker.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Batch size, in total, of all rollout workers and env",
    )
    parser.add_argument(
        "--timestep",
        type=int,
        default=100000,
        help="Total time step to train",
    )
    parser.add_argument(
        "--model",
        type=str,
        default='binary',
        choices=['binary', 'mlp'],
        help="Choose the whether to use mlp or direct parameterization.",
    )
    parser.add_argument(
        "--cp_r_cost_coeff",
        type=float,
        default=0.0,
        help="Loss coefficient for regularization on central planner's reward.",
    )
    parser.add_argument(
        "--cp_r_max",
        type=float,
        default=0.0,
        help="The maximum magnitude of central planner's reward. Default to 1.0 .",
    )
    parser.add_argument(
        "--force_zero_sum",
        type=bool,
        default=False,
        help="Whether to remove the mean of reward. Default is false.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--cp_lr",
        type=float,
        default=None,
        help="Central Planner's learning rate, default is the same as 'lr'.",
    )
    parser.add_argument(
        "--param_assump",
        type=str,
        default='softmax',
        choices=['softmax', 'neural'],
        help="Parameter assumption",
    )
    parser.add_argument(
        "--aware_batch_size",
        type=int,
        default=None,
        help="Batch size for calculation awareness, for saving memory. None means single batch.",
    )

    args = parser.parse_args()
    return args


def main(args):

    ray.init()

    os.environ['RLLIB_NUM_GPUS'] = '1'  # to use gpu

    env_name = 'Matrix_Game_' + args.game_type
    if args.game_type == 'PrisonerDilemma':
        fear, greed = 1.0, 1.0
    elif args.game_type == 'StagHunt':
        fear, greed = 1.0, -1.0
    elif args.game_type == 'Chicken':
        fear, greed = -1.0, 0.5
    else:
        raise ValueError

    register_env(env_name, lambda config: P2M(matrix_game_env_creator(config)))

    env_config = {
        'fear': fear,
        'greed': greed,
    }

    if args.model == 'binary':
        model = {
            'fcnet_hiddens': [],
            'fcnet_activation': 'linear',
        }
        cp_model = {
            'fcnet_hiddens': [],
            'fcnet_activation': 'linear',
        }
    elif args.model == 'mlp':
        model = {
            'fcnet_hiddens': [32, 32],
            'fcnet_activation': 'relu',
        }
        cp_model = {
            'fcnet_hiddens': [128, 128],
            'fcnet_activation': 'relu',
        }
    else:
        raise ValueError

    config = AMDPPOConfig().environment(
        env=env_name,
        env_config=env_config,
    ).training(
        model=model,
        cp_model=cp_model,
        entropy_coeff=0.0,
        train_batch_size=args.batch_size,
        lr=args.lr,
        central_planner_lr=args.cp_lr,
        planner_reward_cost=args.cp_r_cost_coeff,
        gamma=0.99,
        planner_reward_max=args.cp_r_max,
        force_zero_sum=args.force_zero_sum,
        param_assumption=args.param_assump,
        awareness_batch_size=args.aware_batch_size,
        agent_cooperativeness_stats_fn=matrix_game_coop_stats_fn,
    ).rollouts(
        num_rollout_workers=args.num_rollout_workers,
        rollout_fragment_length=args.batch_size,
    ).debugging(log_level="ERROR").framework(framework="torch").resources(
        num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        num_cpus_per_worker=2,
    )

    tune.run(
        AMDPPO,
        name=args.exp_name,
        stop={
            "timesteps_total": args.timestep,
        },
        keep_checkpoints_num=1,
        checkpoint_freq=10,
        local_dir=os.path.join(args.exp_dir, env_name),
        config=config.to_dict(),
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
