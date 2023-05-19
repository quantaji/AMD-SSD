import os
import sys
import argparse

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

from core.algorithms.amd.amd import AMD, AMDConfig
from core.algorithms.amd.wrappers import MultiAgentEnvFromPettingZooParallel as P2M
from core.environments.wolfpack.env import wolfpack_env_creator


class SimpleMLPModelV2(TorchModelV2, nn.Module):

    def __init__(self, obs_space: gym.Space, act_space: gym.Space, num_outputs, *args, **kwargs):
        TorchModelV2.__init__(self, obs_space, act_space, num_outputs, *args, **kwargs)
        nn.Module.__init__(self)

        self.flattened_obs_space = spaces.flatten_space(obs_space)
        self.obs_space = obs_space
        self.action_space = act_space

        self.model = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(self.flattened_obs_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
        )

        self.policy_fn = nn.Linear(32, num_outputs)
        self.value_fn = nn.Linear(32, 1)

    def forward(self, input_dict, state, seq_lens):
        model_out = self.model(input_dict["obs"].to(torch.float32) / 255)
        self._value_out = self.value_fn(model_out)
        return self.policy_fn(model_out), state

    def value_function(self):
        return self._value_out.flatten()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ray rllib training.")
    parser.add_argument(
        "--cuda",
        action="store_true",
        default=False,
        help="Enables GPU training")
    parser.add_argument(
        "--ray-address",
        help="Address of Ray cluster for seamless distributed execution.")
    parser.add_argument(
        "--server-address",
        type=str,
        default=None,
        required=False,
        help="The address of server to connect to if using "
        "Ray Client.")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        required=False,
        help="The number of remote workers used by this algorithm. Default 0, means using local worker.")
    args, _ = parser.parse_known_args()
    
    tmp_dir = os.getenv('ray_temp_dir') or os.path.join(str(os.getenv('SCRATCH')), '.tmp_ray')

    if args.server_address:
        ray.init(f"ray://{args.server_address}", _temp_dir=tmp_dir)
    elif args.ray_address:
        ray.init(address=args.ray_address, _temp_dir=tmp_dir)
    else:
        ray.init(address='auto', _redis_password=os.environ['redis_password'], _temp_dir=tmp_dir)

    # to use gpu
    os.environ['RLLIB_NUM_GPUS'] = '1'
    env_name = 'wolfpack'

    print(args.num_workers)

    register_env(
        env_name,
        lambda config: P2M(wolfpack_env_creator(config)),
    )
    ModelCatalog.register_custom_model("SimpleMLPModelV2", SimpleMLPModelV2)

    config = AMDConfig().environment(
        env=env_name,
        env_config={
            'r_lone': 1.0,
            'r_team': 5.0,
            'r_prey': 0.001,
            'coop_radius': 4,
            'max_cycles': 1024,
        },
        clip_actions=True,
    ).rollouts(
        num_rollout_workers=args.num_workers,
    ).training(
        model={
            "custom_model": "SimpleMLPModelV2",
            "custom_model_config": {},
        },
        train_batch_size=1024,
        lr=2e-5,
        gamma=0.99,
        coop_agent_list=['wolf_1', 'wolf_2'],
        planner_reward_max=0.0,
        force_zero_sum=False,
        param_assumption='neural',
    ).debugging(log_level="ERROR").framework(framework="torch").resources(
        num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        num_cpus_per_worker=1,
    )

    tune.run(
        AMD,
        name="cluster",
        stop={"timesteps_total": 5000000},
        keep_checkpoints_num=3,
        checkpoint_freq=10,
        local_dir= os.path.join(str(os.getenv('SCRATCH')), 'ray_results', env_name),
        config=config.to_dict(),
    )
