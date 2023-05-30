CENTRAL_PLANNER: str = 'central_planner'
STATE_SPACE = 'state_space'
TANH_DETERMINISTIC_DISTRIBUTION = 'tanh_torch_deterministic'
SIGMOID_DETERMINISTIC_DISTRIBUTION = 'sigmoid_torch_deterministic'


class PreLearningProcessing:
    """Constant definition for Sample Batch keywords before feeding to policy to learn"""

    AWARENESS = "agent_awareness"
    R_PLANNER = "reward_by_planner"
    R_PLANNER_CUM = 'cumulative_reward_by_planner'
    TOTAL_ADVANTAGES = 'total_advantages'  # sum of all cooperative agents' advantage, used for computing awareness
    AVAILABILITY = 'availability'  # used for central planner to get only the agent's reward when it is not terminated
    REAL_OBS = 'real_obs'  # in the env, we cannot feed individual actions to planner's observation, therefore, we have to switch obs to real_obs at postprocessing stage.
