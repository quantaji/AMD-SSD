from ..base.constants import BASE_ACTIONS, BASE_GRID_COLOR, BASE_GRID_STATE

WOLFPACK_STATE = BASE_GRID_STATE.copy()
WOLFPACK_STATE.update({
    'U': 'wolf_1',
    'V': 'worf_2',
    'W': 'two wolves at same position',
    'S': 'Self',
    'O': 'the other wolf',
    'P': 'prey',  # pre and wolf will not be at same position, the game will end before that
})
WOLFPACK_AGENT_MAP = {
    'wolf_1': 'U',
    'wolf_2': 'V',
    'prey': 'P',
}
WOLFPACK_AGENT_VIEW_TUNE = {
    'wolf_1': {
        'U': 'S',
        'V': 'O',
        'P': 'P',
    },
    'wolf_2': {
        'U': 'O',
        'V': 'S',
        'P': 'P',
    },
    'prey': {
        'U': 'O',
        'V': 'O',
        'P': 'S',
    },
}
# update for orientation state
WOLFPACK_STATE.update({
    '_': 'empty in orientation',
    '&': 'wall in orientation',
    'u': "wolf_1 in other's orientation",
    'v': "wolf_1 in other's orientation",
    'w': "two wolves orientation",
    's': "self in orientation",
    'o': "the other wolf in orientation",
    'p': "prey in other's orientation",
})
WOLFPACK_ORIENTATION_TUNE = {
    ' ': '_',
    '0': '&',
    '@': '&',
    'U': 'u',
    'V': 'v',
    'W': 'w',
    'S': 's',
    'O': 'o',
    'P': 'p',
}
WOLFPACK_NO_ENTRY_STATE = [
    '@',
    '0',
    '&',
]

WOLFPACK_STATE_SHAPE = (20, 20)
WOLFPACK_OBSERVATION_SHAPE = (16, 21)
WOLFPACK_ORIENTATION_BOUNDING_BOX = {  # x_min, x_max, y_min, y_max offset
    0: [-15, 0, -10, +10],  # up
    1: [-10, +10, 0, +15],  # right
    2: [0, +15, -10, +10],  # down
    3: [-10, +10, -15, 0],  # left
}
WOLFPACK_FPS = 5
WOLFPACK_RESOLUTION = [960, 960]  # [width, hight]
WOLFPACK_ACTIONS = BASE_ACTIONS.copy()

WOLFPACK_COLOR = BASE_GRID_COLOR.copy()
WOLFPACK_COLOR.update({
    'U': [255, 20, 147],  # DeepPink, predator wolf 1
    'V': [238, 130, 238],  # Violet, predator wolf 2
    'W': [255, 0, 0],  # same
    'O': [255, 140, 0],  # Deep orange, for the other wolf
    'S': [0, 128, 0],  # Green for self
    'P': [0, 0, 255],  # Blue, for prey
})
WOLFPACK_COLOR.update({
    '_': [47, 79, 79],  # DarkSlateGray
    '&': [169, 169, 169],  # Dark grey
    'u': [255, 105, 180],  # HotPink
    'v': [221, 160, 221],  # Plum
    'w': [205, 92, 92],  # IndianRed
    's': [60, 179, 113],  # MediumSeaGreen
    'o': [255, 165, 0],  # Orange
    'p': [65, 105, 225],  # Royal blue
})

WOLFPACK_MAP = [
    '@@@@@@@@@@@@@@@@@@@@',
    '@                  @',
    '@      @@    @     @',
    '@@@     @@    @    @',
    '@                  @',
    '@                @ @',
    '@       @@      @  @',
    '@ @@@          @   @',
    '@ @@@              @',
    '@         @@@      @',
    '@    @     @   @   @',
    '@   @@        @@   @',
    '@    @@      @@    @',
    '@                  @',
    '@           @     @@',
    '@   @@     @@@    @@',
    '@  @@@      @@@   @@',
    '@   @              @',
    '@           @      @',
    '@@@@@@@@@@@@@@@@@@@@',
]
