from typing import Any, Dict, List, Tuple

import numpy as np
from gymnasium import spaces
from gymnasium.spaces.utils import flatten, flatten_space
from numpy import ndarray
from pettingzoo.utils.env import ActionDict, ObsDict

from .base import SimpleGameEnv

SELF_PREV_ACT = 'self_prev_action'
OPPO_PREV_ACT = 'opponent_prev_action'
IS_EPS_START = 'is_episode_start'


class MatrixSequential_social_dilemma(SimpleGameEnv):
    """In this game, agent have a chance to see previous observation
    """

    def __init__(
        self,
        payoff_matrix: List[List[List[float]]],
        episode_length: int,
    ):

        self.payoff_matrix: np.ndarray = np.array(payoff_matrix, dtype=float)
        assert self.payoff_matrix.shape == (2, 2, 2)

        super().__init__(
            num_actions=2,
            num_agents=2,
            episode_length=episode_length,
            state_dim=1,  # just a dummy feature to avoid errors. No meaning
        )

        # for each agent it sees itself's action and opponent's action of previous time step
        self.single_observation_space_uf = spaces.Dict({
            SELF_PREV_ACT: spaces.Discrete(self.num_actions),
            SELF_PREV_ACT: spaces.Discrete(self.num_actions),
            IS_EPS_START: spaces.MultiBinary(1),  # whether is is the start of episode, no actions
        })
        self.observation_spaces = dict(zip(
            self.possible_agents,
            [flatten_space(self.single_observation_space_uf)] * len(self.possible_agents),
        ))

        # for state, it records each agent's previous action and bool of start
        self.state_space_uf = spaces.Dict(dict(zip(
            self.possible_agents,
            [spaces.Discrete(self.num_actions)] * len(self.possible_agents),
        )))
        self.state_space_uf[IS_EPS_START] = spaces.MultiBinary(1)
        self.state_space = flatten_space(self.state_space_uf)

        self.last_action: ActionDict = None

        self.opponent_id = {}
        for idx, agent_id in enumerate(self.possible_agents):
            self.opponent_id[agent_id] = self.possible_agents[(idx + 1) % len(self.possible_agents)]

    @property
    def is_eps_start(self) -> bool:
        return self.time_steps == 0

    def initial_state(self) -> ndarray:
        init_state = dict(zip(self.possible_agents, [0] * len(self.possible_agents)))
        init_state[IS_EPS_START] = self.is_eps_start

        return flatten(self.state_space_uf, init_state)

    def observations(self) -> ObsDict:
        if self.is_eps_start:
            obs_uf = {
                SELF_PREV_ACT: 0,
                OPPO_PREV_ACT: 0,
                IS_EPS_START: True,
            }
            obs = flatten(self.single_observation_space_uf, obs_uf)
            return dict(zip(self.possible_agents, [obs] * len(self.possible_agents)))
        else:
            assert self.last_action is not None
            obss = {}
            for idx, agent_id in enumerate(self.possible_agents):
                obs_uf = {
                    SELF_PREV_ACT: self.last_action[agent_id],
                    OPPO_PREV_ACT: self.last_action[self.opponent_id[agent_id]],
                    IS_EPS_START: False,
                }
                obs = flatten(self.single_observation_space_uf, obs_uf)
                obss[agent_id] = obs

            return obss

    def calculate_payoffs(self, actions: ActionDict) -> Dict[str, float]:
        return_list = self.payoff_matrix[actions[self.agents[0]], actions[self.agents[1]]].tolist()
        return dict(zip(self.agents, return_list))

    def step(self, actions: ActionDict) -> Tuple[ObsDict, Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, dict]]:
        self.last_action = actions
        return super().step(actions)


class IteratedPrisonersDilemma(MatrixSequential_social_dilemma):

    def __init__(self, episode_length: int = 10):

        payoff_matrix = [[[-1, -1], [-3, +0]], [[+0, -3], [-2, -2]]]

        super().__init__(payoff_matrix, episode_length)

    def __str__(self) -> str:
        return "Iterated Prisoner Dilemma"


def iterated_prisoner_dilemma_env_creator(config: Dict[str, Any] = {'episode_length': 10}) -> IteratedPrisonersDilemma:
    env = IteratedPrisonersDilemma(episode_length=config['episode_length'])
    return env
