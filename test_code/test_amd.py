import os, sys
import ray
from ray import tune
# from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.pg import PGConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env
from torch import nn
import torch
import gymnasium as gym
import numpy as np
from gymnasium import spaces

module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)
from core.environments.wolfpack.env import wolfpack_env_creator
# from core.algorithms.amd.amd_torch_policy import AMDAgentTorchPolicy
from core.algorithms.amd.amd import AMDConfig, AMD
from core.algorithms.amd.wrappers import ParallelEnvWithCentralPlanner as PEnvCP, MultiAgentEnvFromPettingZooParallel as P2M, MultiAgentEnvWithCentralPlanner as MEnvCP
from ray.rllib.policy.policy import PolicySpec


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
        # print('-----*****-----*****-----', self.obs_total_dim, input_dict["obs"].flatten(start_dim=1, end_dim=-1).shape)
        # model_out = self.model(input_dict["obs"].permute(0, 3, 1, 2))

        # print(self.name, self.obs_space, self.action_space, input_dict.keys())

        model_out = self.model(input_dict["obs"].to(torch.float32) / 255)
        self._value_out = self.value_fn(model_out)
        return self.policy_fn(model_out), state

    def value_function(self):
        return self._value_out.flatten()


if __name__ == "__main__":
    ray.init()

    # to use gpu
    os.environ['RLLIB_NUM_GPUS'] = '1'
    env_name = 'wolfpack'

    register_env(env_name, lambda config: P2M(wolfpack_env_creator(config)))
    ModelCatalog.register_custom_model("SimpleMLPModelV2", SimpleMLPModelV2)

    config = AMDConfig().environment(
        env=env_name,
        env_config={
            'r_lone': 1.0,
            'r_team': 5.0,
            'r_prey': 0.0,
            'max_cycles': 1024,
        },
        clip_actions=True,
    ).rollouts(num_rollout_workers=4, rollout_fragment_length=128).training(
        model={
            "custom_model": "SimpleMLPModelV2",
            # Extra kwargs to be passed to your model's c'tor.
            "custom_model_config": {},
        },
        train_batch_size=512,
        lr=2e-5,
        gamma=0.99,
        coop_agent_list=['wolf_1', 'wolf_2'],
        planner_reward_max=0.5,
        force_zero_sum=True,
        param_assumption='neural',
    ).debugging(log_level="ERROR", ).framework(framework="torch", ).resources(
        num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        num_cpus_per_worker=3,
    )

    tune.run(
        AMD,
        name="no_planner_run",
        stop={"timesteps_total": 5000000},
        checkpoint_freq=10,
        local_dir="~/ray_results/" + env_name,
        config=config.to_dict(),
    )
