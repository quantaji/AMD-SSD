import os
import sys

import gymnasium as gym
import numpy as np
import ray
import supersuit as ss
import torch
from ray import tune
from ray.tune.registry import register_env
from torch import nn

module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)

from core.algorithms.amd.wrappers import \
    MultiAgentEnvFromPettingZooParallel as P2M
from core.environments.wolfpack import wolfpack_env_creator, wolfpack_coop_stats_fn
from core.algorithms.amd_ppo import AMDPPO, AMDPPOConfig

if __name__ == "__main__":
    ray.init()

    # to use gpu
    os.environ['RLLIB_NUM_GPUS'] = '1'

    # real setting
    n_env = 16 
    n_worker = 6
    length = 1024

    # # test setting
    # n_env = 1
    # n_worker = 0
    # length = 512

    batch_size = n_env * max(1, n_worker) * length

    env_name = 'wolfpack'
    config_template = {
        'r_lone': 1.0,
        'r_team': 5.0,
        'r_prey': 0.01,
        'r_starv': -0.01,
        'max_cycles': length,
    }

    register_env(env_name, lambda config: P2M(ss.normalize_obs_v0(ss.dtype_v0(
        wolfpack_env_creator(config),
        np.float32,
    ))))

    config = AMDPPOConfig().environment(
        env=env_name,
        env_config=config_template,
    ).rollouts(
        num_rollout_workers=n_worker,
        # num_rollout_workers=0,
        rollout_fragment_length=length,
        num_envs_per_worker=n_env,
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
        # model={
        #     "conv_filters": [  # 16x21x3
        #         [16, [4, 4], 2],  # 7x9x16
        #         [32, [4, 4], 2],  # 2x3x32
        #         [256, [4, 6], 1],
        #     ],
        #     'fcnet_hiddens': [32, 32],
        # },
        cp_model={
            'fcnet_hiddens': [256, 128],
            'fcnet_activation': 'relu',
            "use_lstm": True,
            "lstm_cell_size": 32,
            "max_seq_len": 32,
        },
        train_batch_size=batch_size,
        lr=1e-4,
        lr_schedule=[[0, 0.00136], [20000000, 0.000028]],
        gamma=0.99,
        lambda_=0.95,
        entropy_coeff=0.000687,
        vf_loss_coeff=0.5,
        sgd_minibatch_size=batch_size // 4,
        num_sgd_iter=10,
        agent_pseudo_lr=1e-4,
        central_planner_lr=1e-4,
        coop_agent_list=['wolf_1', 'wolf_2'],
        # planner_reward_max=0.05,
        planner_reward_max=0.2,
        # planner_reward_max=0.0,
        reward_distribution='tanh',
        force_zero_sum=False,
        # param_assumption='neural',
        awareness_batch_size=1024,
        # neural_awareness_method='grad',
        param_assumption='softmax',
        use_cum_reward=False,
        # use_cum_reward=True,
        pfactor_half_step=2 * (10**6) * 8,
        pfactor_step_scale=5 * (10**5),
        agent_cooperativeness_stats_fn=lambda sample_batch: wolfpack_coop_stats_fn(
            sample_batch=sample_batch,
            coop_reward=config_template['r_team'],
        ),
    ).debugging(
        log_level="ERROR",
        seed=132561,
    ).framework(framework="torch", ).resources(
        num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        num_cpus_per_worker=1,
    )

    tune.run(
        AMDPPO,
        name="AMDPPO-with-central-planner-r_max=0.2-adv-reward-large-batch-size",
        # name="AMDPPO-no-central-planner-retry-1",
        stop={"timesteps_total": 60 * (10**6)},
        keep_checkpoints_num=3,
        checkpoint_freq=10,
        # local_dir="~/ray_experiment_results/" + env_name,
        # local_dir="~/ray_test/" + env_name,
        local_dir="~/ray_debug/" + env_name,
        config=config.to_dict(),
        # resume="LOCAL+ERRORED",
    )
