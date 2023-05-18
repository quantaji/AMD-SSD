import os, sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from core.environments.utils import ascii_list_to_array, ascii_array_to_rgb_array, ascii_array_to_str, ascii_dict_to_color_array
from core.environments.gathering.env import Gathering, GatheringEnv
import numpy as np
from gymnasium.utils import seeding
from PIL import Image
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from pettingzoo.butterfly import pistonball_v5
import supersuit as ss
import torch
from torch import nn
from gymnasium import spaces

class CNNModelV2(TorchModelV2, nn.Module):
    def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
        TorchModelV2.__init__(self, obs_space, act_space, num_outputs, *args, **kwargs)
        nn.Module.__init__(self)

        self.flatten_obs_space = spaces.flatten_space(obs_space)
        self.obs_space = obs_space

        self.model = nn.Sequential(
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Linear(self.flattened_obs_space.shape[0], 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        )
        self.policy_fn = nn.Linear(512, num_outputs)
        self.value_fn = nn.Linear(512, 1)

    def forward(self, input_dict, state, seq_lens):
        model_out = self.model(input_dict["obs"].to(torch.float32))
        self._value_out = self.value_fn(model_out)
        return self.policy_fn(model_out), state

    def value_function(self):
        return self._value_out.flatten()

def env_creator(args):
    env = GatheringEnv()
    # env = ss.color_reduction_v0(env, mode='B')
    env = ss.dtype_v0(env, 'float32')
    # env = ss.resize_v0(env, x_size=84, y_size=84)
    # env = ss.frame_stack_v1(env, 3)
    env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
    return env

    