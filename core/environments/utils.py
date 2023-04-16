import numpy as np
from typing import Dict, Tuple


def ascii_list_to_array(ascii_list):
    """converts a list of strings into a numpy array

    Parameters
    ----------
    ascii_list: list of strings
        List describing what the map should look like
    Returns
    -------
    arr: np.ndarray
        numpy array describing the map with ' ' indicating an empty space
    """

    arr = np.full((len(ascii_list), len(ascii_list[0])), ' ')
    for row in range(arr.shape[0]):
        for col in range(arr.shape[1]):
            arr[row, col] = ascii_list[row][col]
    return arr


def ascii_array_to_rgb_array(ascii_arr: np.ndarray, ascii_color_dict: Dict[str, Tuple[int, int, int]]) -> np.ndarray:
    assert len(ascii_arr.shape) == 2, "ascii_arr must have shape of (H, W)!"

    return np.array([[ascii_color_dict[ascii_arr[i][j]] for j in range(ascii_arr.shape[1])] for i in range(ascii_arr.shape[0])], dtype=np.uint8)


def ascii_array_to_str(ascii_arr: np.ndarray) -> str:
    return '\n'.join([' '.join(line_arr) for line_arr in ascii_arr.tolist()])
