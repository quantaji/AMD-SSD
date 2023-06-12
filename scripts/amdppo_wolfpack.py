import argparse
import os
import sys
from pathlib import Path

import numpy as np
import ray
import supersuit as ss
from ray import tune
from ray.tune.registry import register_env

module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)

from core.algorithms.amd.wrappers import MultiAgentEnvFromPettingZooParallel as P2M
from core.algorithms.amd_ppo import AMDPPO, AMDPPOConfig
from core.environments.wolfpack import wolfpack_coop_stats_fn, wolfpack_env_creator


def parse_args():
    parser = argparse.ArgumentParser("Adaptive Mechanism Design on Matrix Game")

    parser.add_argument(
        "--debug",
        type=bool,
        help="If use debug, use a smaller setting, and use another directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=114514,
        help="Random seed",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        help="Name of this experiment.",
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        default=os.path.join(os.environ.get('SCRATCH', str(Path.home())), 'forl-exp'),
        help="Path to store experiment results. The folder of this experiment is 'exp_dir/env_name/exp_name/'",
    )
    parser.add_argument(
        "--num_rollout_workers",
        type=int,
        default=6,
        help="Number of rollout works. Default is 0, meaning only using local worker.",
    )
    parser.add_argument(
        "--num_env",
        type=int,
        default=16,
        help="Number of environment.",
    )
    parser.add_argument(
        "--env_max_len",
        type=int,
        default=1024,
        help="maximum time step in one episode emulation of env.",
    )
    parser.add_argument(
        "--timestep",
        type=int,
        default=14155776,
        help="Total time step to train",
    )
    parser.add_argument(
        "--model",
        type=str,
        default='conv',
        choices=['conv', 'lstm'],
        help="Choose the whether to use pure convolution network, or conv with lstm.",
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
        default=768,
        help="Batch size for calculation awareness, for saving memory. None means single batch.",
    )
    parser.add_argument(
        "--amd_schedule_half",
        type=int,
        default=2 * (10**6) * 8,
        help="Schedule factor determine when amd loss's factor is 0.5.",
    )
    parser.add_argument(
        "--amd_schedule_scale",
        type=int,
        default=5 * (10**5),
        help="Schedule factor determine increasing speed.",
    )
    parser.add_argument(
        "--cum_reward",
        type=bool,
        default=False,
        help="Whether to use cumulated reward (original amd paper), or q-value adjustment.",
    )
    parser.add_argument(
        "--ray-address",
        help="Address of Ray cluster for seamless distributed execution.",
    )
    parser.add_argument(
        "--server-address",
        type=str,
        default=None,
        required=False,
        help="The address of server to connect to if using Ray Client.",
    )
    args = parser.parse_args()
    return args


def main(args):

    ray.init(
        address='auto',
        _redis_password=os.environ.get('redis_password', ""),
    )

    os.environ['RLLIB_NUM_GPUS'] = '1'  # to use gpu

    n_env = 1 if args.debug else args.num_env
    n_worker = 0 if args.debug else args.num_rollout_workers
    length = 256 if args.debug else args.env_max_len

    batch_size = n_env * max(1, n_worker) * length
    aware_bs = max(length, args.aware_batch_size) if args.aware_batch_size else None

    env_name = 'wolfpack'
    register_env(env_name, lambda config: P2M(ss.normalize_obs_v0(ss.dtype_v0(
        wolfpack_env_creator(config),
        np.float32,
    ))))
    env_config = {
        'r_lone': 1.0,
        'r_team': 5.0,
        'r_prey': 0.01,
        'r_starv': -0.05,
        'coop_radius': 6,
        'max_cycles': length,
    }

    # model config
    if args.model == 'lstm':
        model = {
            "conv_filters": [
                [6, [3, 3], 1],
            ],
            "post_fcnet_hiddens": [32, 32],
            "use_lstm": True,
            "lstm_cell_size": 128,
            "max_seq_len": 32,
        }
    elif args.model == 'conv':
        model = {
            "conv_filters": [  # 16x21x3
                [16, [4, 4], 2],  # 7x9x16
                [32, [4, 4], 2],  # 2x3x32
                [64, [4, 6], 1],
            ],
            'fcnet_hiddens': [32, 32],
        }
    else:
        raise ValueError
    cp_model = {
        'fcnet_hiddens': [256, 128],
        'fcnet_activation': 'relu',
        "use_lstm": True,
        "lstm_cell_size": 32,
        "max_seq_len": 32,
    }

    config = AMDPPOConfig().environment(
        env=env_name,
        env_config=env_config,
    ).training(
        model=model,
        cp_model=cp_model,
        entropy_coeff=0.000687,
        vf_loss_coeff=0.5,
        gamma=0.99,
        lambda_=0.95,
        train_batch_size=batch_size,
        sgd_minibatch_size=batch_size // 4,
        num_sgd_iter=10,
        lr=args.lr,
        central_planner_lr=args.cp_lr,
        coop_agent_list=['wolf_1', 'wolf_2'],
        planner_reward_cost=args.cp_r_cost_coeff,
        planner_reward_max=args.cp_r_max,
        force_zero_sum=args.force_zero_sum,
        param_assumption=args.param_assump,
        awareness_batch_size=aware_bs,
        use_cum_reward=args.cum_reward,
        pfactor_half_step=args.amd_schedule_half,
        pfactor_step_scale=args.amd_schedule_scale,
        agent_cooperativeness_stats_fn=lambda sample_batch: wolfpack_coop_stats_fn(
            sample_batch=sample_batch,
            coop_reward=env_config['r_team'],
        ),
    ).rollouts(
        num_rollout_workers=n_worker,
        rollout_fragment_length=length,
        num_envs_per_worker=n_env,
    ).debugging(
        log_level="ERROR",
        seed=args.seed,
    ).framework(framework="torch").resources(
        num_gpus=1,
        num_cpus_per_worker=1,
    )

    if not args.debug:
        local_dir = os.path.join(args.exp_dir, env_name)
    else:
        local_dir = os.path.join(os.environ.get('SCRATCH', str(Path.home())), 'ray_debug', env_name)

    print("local dir: ", local_dir)

    tune.run(
        AMDPPO,
        name=args.exp_name,
        stop={
            "timesteps_total": args.timestep,
        },
        keep_checkpoints_num=1,
        checkpoint_freq=4,
        local_dir=local_dir,
        config=config.to_dict(),
        resume="LOCAL+ERRORED",
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
