from abc import ABC
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from gymnasium import spaces
from gymnasium.utils import seeding
from gymnasium.vector.utils import create_empty_array
from pettingzoo.utils.env import ActionDict, AgentID, ObsDict, ParallelEnv
from ray.rllib.policy.sample_batch import SampleBatch

from ...algorithms.amd.constants import PreLearningProcessing
from .base import SimpleGameEnv


class MatrixGameEnv(SimpleGameEnv):

    def __init__(
        self,
        fear: float,
        greed: float,
    ):
        self.fear = fear
        self.greed = greed

        self.R = 3
        self.P = 1
        self.T = self.R + self.greed
        self.S = self.P - self.fear

        num_actions = 2
        num_agents = 2
        state_dim = 1  # just a dummy feature to avoid errors. No meaning
        episode_length = 1

        super().__init__(
            num_actions=num_actions,
            num_agents=num_agents,
            episode_length=episode_length,
            state_dim=state_dim,
        )

        # init observation space and state space
        self.state_space = spaces.Discrete(1)
        self.observation_spaces = dict(zip(self.possible_agents, [self.state_space] * len(self.possible_agents)))

    def __str__(self):
        description = "Matrix_Game_Greed=" + str(self.greed) + "_Fear=" + str(self.fear)
        return description

    def initial_state(self):
        return np.array(0)  #dummy feature

    def calculate_payoffs(self, actions):
        action_list = [actions[agent_id] for agent_id in self.agents]
        return_list = None

        if action_list[0] == 1:
            if action_list[1] == 0:
                return_list = [self.S, self.T]
            else:
                return_list = [self.R, self.R]
        else:
            if action_list[1] == 0:
                return_list = [self.P, self.P]
            else:
                return_list = [self.T, self.S]

        return dict(zip(self.agents, return_list))


def matrix_game_env_creator(config: Dict[str, Any] = {
    'fear': 1.0,
    'greed': 1.0,
}) -> MatrixGameEnv:
    env = MatrixGameEnv(
        fear=config['fear'],
        greed=config['greed'],
    )
    return env


def coop_stats_fn(sample_batch: SampleBatch) -> float:
    actions = sample_batch[SampleBatch.ACTIONS]
    if isinstance(actions, torch.Tensor):
        actions = actions.detach().cpu().numpy()

    return (actions == 1).mean()
