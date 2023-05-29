import os
import sys
import time

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
from core.algorithms.amd.wrappers import \
    MultiAgentEnvFromPettingZooParallel as P2M
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

    algo = config.build().from_checkpoint(
        # '/home/quanta/ray_results_new/wolfpack/PPO/PPO_wolfpack_0e5b8_00000_0_2023-05-21_21-16-58/checkpoint_002500'
        '/home/quanta/ray_experiment_results/wolfpack/PPO-lstm-setting-3/PPO_wolfpack_e73c7_00000_0_2023-05-28_15-45-27/checkpoint_004880', )

    worker = algo.workers.local_worker()
    policy_map = worker.policy_map
    policy_mapping_fn = worker.policy_mapping_fn
    for policy_id in policy_map.keys():
        print(policy_id)
    env = P2M(wolfpack_env_creator({
        'r_lone': 1.0,
        'r_team': 5.0,
        'r_prey': 0.0,
        'coop_radius': 4,
        'max_cycles': 1000,
    }))
    obs, _ = env.reset()
    terminated = truncated = False

    timestep = 0

    env.par_env.env.render_mode = 'human'  # 'human'

    env.render()
    render_result = []

    while not terminated and not truncated:

        action = {}
        for agent_id in obs.keys():
            action[agent_id] = algo.compute_single_action(
                observation=obs[agent_id],
                policy_id=policy_mapping_fn(agent_id),
                # explore=False,
            )
        # action = algo.compute_single_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        terminated = terminated['__all__']
        truncated = truncated['__all__']
        timestep += 1

        if env.par_env.env.render_mode == 'human':
            time.sleep(0.2)
            env.render()
        else:
            render_img = env.render()
            render_result.append(render_img)
