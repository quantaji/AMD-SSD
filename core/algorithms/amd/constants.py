CENTRAL_PLANNER: str = 'central_planner'
STATE_SPACE = 'state_space'


class PreLearningProcessing:
    """Constant definition for Sample Batch keywords before feeding to policy to learn"""

    AWARENESS = "agent_awareness"
    R_PLANNER = "reward_by_planner"
    AVAILABILITY = 'availability'  # used for central planner to get only the agent's reward when it is not terminated
