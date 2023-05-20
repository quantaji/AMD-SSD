CENTRAL_PLANNER: str = 'central_planner'
STATE_SPACE = 'state_space'
TANH_DETERMINISTIC_DISTRIBUTION = 'tanh_torch_deterministic'
DETERMINISTIC_DISTRIBUTION = 'torch_deterministic'


class PreLearningProcessing:
    """Constant definition for Sample Batch keywords before feeding to policy to learn"""

    AWARENESS = "agent_awareness"
    R_PLANNER = "reward_by_planner"
    R_PLANNER_CUM = 'cumulative_reward_by_planner'
    TOTAL_ADVANTAGES = 'total_advantages'  # sum of all cooperative agents' advantage, used for computing awareness
    AVAILABILITY = 'availability'  # used for central planner to get only the agent's reward when it is not terminated
    DISCOUNTED_FACTOR_MATRIX = 'discounted_factor_matrix'  # used for processing a batch, a btach may have multiple episodes, this matrix is TxT indicating which time appear in previous timesteps for calculation of discounted cumsum
