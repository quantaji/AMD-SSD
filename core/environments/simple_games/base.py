from abc import ABC
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.utils import seeding
from numpy import ndarray
from pettingzoo.utils.env import ActionDict, AgentID, ObsDict, ParallelEnv


class SimpleGameEnv(ParallelEnv):
    """Base environment for matrix game, coin game, etc
    """

    def __init__(
        self,
        num_actions: int,
        num_agents: int,
        episode_length: int,
        state_dim: int,
    ):
        self.num_actions: int = num_actions
        self.n_agents: int = num_agents
        self.state_dim: int = state_dim
        self.episode_length: int = episode_length

        # initialize constants requied for pettingzoo
        self.possible_agents = ['agent_{}'.format(i) for i in range(self.n_agents)]
        self.agents = self.possible_agents[:]

        # NOTE: observation space and state space is left subclass to define, because this may differ across games
        single_agent_action_space = spaces.Discrete(self.num_actions)
        self.action_spaces = dict(zip(self.possible_agents, [single_agent_action_space] * len(self.possible_agents)))

        self.state_space: spaces.Space

        # a temporary variable for storing state
        self._state: ndarray = None
        # counter
        self.time_steps: int = None

    def state(self) -> ndarray:
        return self._state

    def observations(self) -> ObsDict:
        """This game is very simple, every observation is the same as state in this assumtion.
        By default, all agetn sees the same state, this can be overidden.
        """
        single_agent_observation = self.state()
        return dict(zip(self.possible_agents, [single_agent_observation] * len(self.possible_agents)))

    def initial_state(self) -> ndarray:
        raise NotImplementedError

    def calculate_payoffs(self, actions: ActionDict) -> Dict[str, float]:
        raise NotImplementedError

    def seed(self, seed=None):
        self.randomizer, seed = seeding.np_random(seed)

    def reset(
        self,
        seed: int | None = None,
        return_info: bool = False,
        options: dict | None = None,
    ) -> ObsDict:

        if seed is not None:
            self.seed(seed=seed)

        self.agents = self.possible_agents[:]
        self._state = self.initial_state()
        self.time_steps = 0

        if return_info:
            return self.observations(), {}
        else:
            return self.observations()

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def step(self, actions: ActionDict) -> Tuple[
            ObsDict,
            Dict[str, float],
            Dict[str, bool],
            Dict[str, bool],
            Dict[str, dict],
    ]:

        self.time_steps += 1

        obs = self.observations()
        rwd = self.calculate_payoffs(actions)
        term = dict(zip(self.possible_agents, [self.time_steps >= self.episode_length] * len(self.possible_agents)))
        trunc = dict(zip(self.possible_agents, [False] * len(self.possible_agents)))  # never truncates
        info = dict(zip(self.possible_agents, [{}] * len(self.possible_agents)))

        # also, when terminates, agent list set to []
        if self.time_steps >= self.episode_length:
            self.agents = []

        return obs, rwd, term, trunc, info
