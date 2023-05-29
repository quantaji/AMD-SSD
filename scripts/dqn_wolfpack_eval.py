import os
import sys
import time

import gymnasium as gym
import numpy as np
import ray
import torch
from gymnasium import spaces
from ray import tune
from ray.rllib.algorithms.dqn import DQN, DQNConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.exploration.epsilon_greedy import EpsilonGreedy
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

    register_env(
        env_name,
        lambda config: P2M(wolfpack_env_creator(config)),
    )

    config = DQNConfig().multi_agent(
        policies_to_train=[],
        policies=['predator', 'prey'],
        policy_mapping_fn=(lambda agent_id, *args, **kwargs: {
            'wolf_1': 'predator',
            'wolf_2': 'predator',
            'prey': 'prey',
        }[agent_id]),
    ).environment(
        env=env_name,
        env_config={
            'r_lone': 1.0,
            'r_team': 5.0,
            'r_prey': 0.0,
            'r_starv': -0.01,
            'coop_radius': 4,
            'max_cycles': 1000,
            'render_mode': 'rgb_array',
        },
        clip_actions=True,
    ).rollouts(
        num_rollout_workers=4,
        rollout_fragment_length=128,
    ).evaluation(
        # evaluation_num_workers=2,
        evaluation_interval=1,
        # evaluation_parallel_to_training=True,
        evaluation_duration=10,  # default unit is episodes
        evaluation_config=DQNConfig.overrides(
            env_config={
                # "record_env": "videos",  # folder to store video?
                "render_env": '~/video',
            }, ),
    ).training(
        model={
            "conv_filters": [  # 16x21x3
                [16, [4, 4], 2],  # 7x9x16
                [32, [4, 4], 2],  # 2x3x32
                [256, [4, 6], 1],
            ],
        },
        train_batch_size=1000,
        lr=2e-5,
        gamma=0.99,
        v_min=0.0,
        v_max=10.0,
        # double_q=True,
    ).framework(framework="torch").resources(
        num_gpus=1,
        num_cpus_per_worker=3,
    )

    algo = config.build().from_checkpoint('/home/quanta/ray_experiment_results/wolfpack/dqn-with-living-panelty-dual/DQN_wolfpack_d2542_00000_0_2023-05-30_00-11-55/checkpoint_000077')

    worker = algo.workers.local_worker()
    policy_map = worker.policy_map
    policy_mapping_fn = worker.policy_mapping_fn
    for policy_id in policy_map.keys():
        print(policy_id)
    env = P2M(wolfpack_env_creator({
        'r_lone': 1.0,
        'r_team': 5.0,
        'r_prey': 0.0,
·        'r_starv': -0.01,
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
