from typing import Dict, Tuple

import numpy as np


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

    arr = np.full((len(ascii_list), len(ascii_list[0])), ' ', dtype="|U1")
    for row in range(arr.shape[0]):
        for col in range(arr.shape[1]):
            arr[row, col] = ascii_list[row][col]
    return arr


def ascii_dict_to_color_array(ascii_color_dict: Dict[str, Tuple[int, int, int]]) -> Dict[int, np.ndarray]:
    color_array = np.full((2**8, 3), 0, dtype=np.uint8)
    for state_chr in ascii_color_dict.keys():
        color_array[ord(state_chr)] = np.array(ascii_color_dict[state_chr])

    return color_array


def ascii_array_to_rgb_array(ascii_arr: np.ndarray, color_arr: np.ndarray) -> np.ndarray:
    assert len(ascii_arr.shape) == 2, "ascii_arr must have shape of (H, W)!"
    assert color_arr.shape == (2**8, 3), "color_arr must have shape of (256, 3)!"

    return color_arr[ascii_arr.astype("|S1").view(np.uint8)]


def ascii_array_to_rgb_array_deprecated(ascii_arr: np.ndarray, ascii_color_dict: Dict[str, Tuple[int, int, int]]) -> np.ndarray:
    assert len(ascii_arr.shape) == 2, "ascii_arr must have shape of (H, W)!"

    return np.array([[ascii_color_dict[ascii_arr[i][j]] for j in range(ascii_arr.shape[1])] for i in range(ascii_arr.shape[0])], dtype=np.uint8)


def ascii_array_to_str(ascii_arr: np.ndarray) -> str:
    arr = None
    if not ascii_arr.dtype.char == 'U':
        arr = ascii_arr.astype('|U1')
    else:
        arr = ascii_arr
    return '\n'.join([' '.join(line_arr) for line_arr in arr.tolist()])
