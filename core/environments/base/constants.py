import numpy as np

ORIENTATIONS = {
    0: 'up',
    1: 'right',
    2: 'down',
    3: 'left',
}

ORIENTATION_CHANGE = {
    0: np.array([-1, 0]),
    1: np.array([0, +1]),
    2: np.array([+1, 0]),
    3: np.array([0, -1]),
}

BASE_ACTIONS = {
    0: 'move_left',
    1: 'move_right',
    2: 'move_up',
    3: 'move_down',
    4: 'stay',
    5: 'turn_clockwise',
    6: 'turn_counterclockwise',
}

ACTION_POSITION_CHANGE = {
    #0: np.array([0, -1]),
    #1: np.array([0, +1]),
    #2: np.array([-1, 0]),
    #3: np.array([+1, 0]),
    0: np.array([-1, 0]),
    1: np.array([0, +1]),
    2: np.array([+1, 0]),
    3: np.array([0, -1]),
}

ACTION_ORIENTATION_CHANGE = {
    5: +1,
    6: -1,
}

BASE_GRID_STATE = {
    ' ': 'empty',
    '@': 'wall',
    '0': 'void',  # void for observation that is outside the grid
}

BASE_GRID_COLOR = {
    ' ': [0, 0, 0],  # Black, background
    '0': [0, 0, 0],  # Black, background beyond map walls
    '@': [128, 128, 128],  # Grey, board walls
}
