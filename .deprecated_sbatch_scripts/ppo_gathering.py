import os
import sys
import argparse

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

from core.environments.gathering.env import gathering_env_creator

if __name__ == "__main__":
    ## To use in slurm 
    parser = argparse.ArgumentParser(description="Ray rllib training.")
    parser.add_argument("--cuda", action="store_true", default=False, help="Enables GPU training")
    parser.add_argument("--ray-address", help="Address of Ray cluster for seamless distributed execution.")
    parser.add_argument("--server-address", type=str, default=None, required=False, help="The address of server to connect to if using "
                        "Ray Client.")
    parser.add_argument("--num_workers", type=int, default=4, required=False, help="The number of remote workers used by this algorithm. Default 0, means using local worker.")
    args, _ = parser.parse_known_args()

    tmp_dir = os.getenv('ray_temp_dir') or os.path.join(str(os.getenv('SCRATCH')), '.tmp_ray')

    if args.server_address:
        ray.init(f"ray://{args.server_address}", _temp_dir=tmp_dir)
    elif args.ray_address:
        ray.init(address=args.ray_address, _temp_dir=tmp_dir)
    else:
        ray.init(address='auto', _redis_password=os.environ['redis_password'], _temp_dir=tmp_dir)

    #ray.init()

    # to use gpu
    os.environ['RLLIB_NUM_GPUS'] = '1'

    env_name = 'gathering'

    print('number of workers: ', args.num_workers)
    print('number of gpus: ', int(os.environ.get("RLLIB_NUM_GPUS", "0")))

    config_template = {
        'apple_respawn': 3,
        'apple_number': 5,
        'player_blood': 1,
        'tagged_time_number': 5,
        'max_cycles': 1024,
        'r_starv': -0.01,
    }

    register_env(env_name, lambda config: ParallelPettingZooEnv(gathering_env_creator(config)))

    config = PPOConfig().environment(
        env=env_name,
        env_config=config_template,
    ).rollouts(num_rollout_workers=4, rollout_fragment_length=128).training(
        # model={
        #     "conv_filters": [  # 16x21x3
        #         [16, [4, 4], 2],  # 7x9x16
        #         [32, [4, 4], 2],  # 2x3x32
        #         [256, [4, 6], 1],
        #     ],
        # },
        # model={
        #     "conv_filters": [  # 16x21x3
        #         [8, [4, 4], 1],  # 7x9x16
        #     ],
        #     "post_fcnet_hiddens": [32, 32],
        #     "use_lstm": True,
        #     "lstm_cell_size": 128,
        #     "max_seq_len": 32,
        # },
        model={
            "conv_filters": [  # 16x21x3
                [16, [4, 4], 2],  # 7x9x16
                [32, [4, 4], 2],  # 2x3x32
                [64, [4, 6], 1],
            ],
            "post_fcnet_hiddens": [128],
            "use_lstm": True,
            "lstm_cell_size": 128,
            "max_seq_len": 32,
        },
        # train_batch_size=16384,
        train_batch_size=8192,
        lr=1e-4,
        gamma=0.99,
        lambda_=0.9,
        use_gae=True,
        clip_param=0.4,
        grad_clip=None,
        entropy_coeff=0.1,
        vf_loss_coeff=0.25,
        sgd_minibatch_size=1024,
        num_sgd_iter=10,
    ).debugging(log_level="ERROR", ).framework(framework="torch", ).resources(
        num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        num_cpus_per_worker=3,
    ).multi_agent(
        policies=['predator'],
        policy_mapping_fn=(lambda agent_id, *args, **kwargs: {
            'blue_p': 'predator',
            'red_p': 'predator',
        }[agent_id]),
    )

    tune.run(
        "PPO",
        name="PPO-gathering_cluster",
        stop={"timesteps_total": 5000000},
        keep_checkpoints_num=3,
        checkpoint_freq=10,
        local_dir=os.path.join(str(os.getenv('SCRATCH')), 'ray_results', env_name),
        # local_dir="~/ray_test/" + env_name,
        config=config.to_dict(),
    )
