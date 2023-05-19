from ..base.constants import BASE_ACTIONS, BASE_GRID_COLOR, BASE_GRID_STATE

WOLFPACK_STATE = BASE_GRID_STATE.copy()
WOLFPACK_STATE.update({
    'U': 'wolf_1',
    'V': 'worf_2',
    'W': 'two wolves at same position',
    'S': 'Self',
    'O': 'the other wolf',  # or team mates
    'X': 'opponent',  # for wolf, opponent is prey, for prey, opponent is wolf
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
        'P': 'X',
    },
    'wolf_2': {
        'U': 'O',
        'V': 'S',
        'P': 'X',
    },
    'prey': {
        'U': 'X',
        'V': 'X',
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
    'x': "opponents in other's orientation",
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
    'X': 'x',
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
    'U': [205, 92, 92],  # wolf 1, indian red
    'V': [255, 140, 0],  # wolf 2, dark orange
    'W': [0, 128, 0],  # two wolf, green
    'O': [173, 216, 230],  # teammates, light blue
    'S': [0, 0, 255],  # self, blue
    'P': [0, 0, 255],  # prey, blue
    'X': [255, 0, 0],  # opponent, red
})
WOLFPACK_COLOR.update({
    '_': [47, 79, 79],  # DarkSlateGray
    '&': [169, 169, 169],  # Dark grey
    'u': [220, 20, 60],  # Crimson
    'v': [255, 165, 0],  # Orange
    'w': [50, 205, 50],  # Limegreen
    's': [65, 105, 225],  # Royal blue
    'o': [176, 196, 222],  # light steel blue
    'p': [65, 105, 225],  # Royal blue
    'x': [139, 0, 0],  # dark red
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
