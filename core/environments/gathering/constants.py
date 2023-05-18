from ..base.constants import BASE_GRID_STATE, BASE_GRID_COLOR, BASE_ACTIONS, ORIENTATION_CHANGE

GATHERING_STATE = BASE_GRID_STATE.copy()
GATHERING_STATE.update({
    'U': 'blue_p',
    'V': 'red_p',
    'W': 'two players at same position',
    'S': 'agent self',
    'O': 'opponent',
    'P': 'apple',
    'F': 'beam and player',
    'B': 'beam',
    'C': 'apple and player'
})
GATHERING_AGENT_MAP = {
    'blue_p': 'U', 
    'red_p': 'V',
    'apple': 'P',
}
GATHERING_AGENT_VIEW_TUNE = {
    'blue_p': {
        'U': 'S',
        'V': 'O',
        'P': 'P',
    },
    'red_p': {
        'U': 'O',
        'V': 'S',
        'P': 'P',
    },
    ## perhaps change here
    'apple': {
        'U': 'O',
        'V': 'O',
        'P': 'S',
    },
}

GATHERING_ORIENTATION_CHANGE = ORIENTATION_CHANGE.copy()

# update for orientation state
GATHERING_STATE.update({
    '_': 'empty in orientation',
    '&': 'wall in orientation',
    'u': "blue_p in other's orientation",
    'v': "red_p in other's orientation",
    'w': "two players orientation",
    's': "self in orientation",
    'o': "the opponent in orientation",
    'p': "apple in other's orientation",
    'f': "beam and player",
    'b': 'beam in orientation',
    'c': 'apple and player'
})
GATHERING_ORIENTATION_TUNE = {
    ' ': '_', #empty
    '0': '&', #void
    '@': '&', #wall
    'U': 'u',
    'V': 'v',
    'W': 'w',
    'S': 's',
    'O': 'o',
    'P': 'p',
    'F': 'f',
    'B': 'b',
    'C': 'c',
}
GATHERING_NO_ENTRY_STATE = [
    '@', 
    '0',
    '&',
]

GATHERING_APPLE_NO_ENTRY_STATE = [
    '@',
    '0',
    '&',
    'U',
    'V',
    'W',
    'S',
    'O',
]
## For display overlapping of beam and player(F) 
# and player and apple -> 'C'; perhaps not used
GATHERING_BEAM_PLAYER_STATE = [ 
    'U',
    'V',
    'W',
    'S',
    'O',
]

GATHERING_OBSERVATION_SHAPE = (11, 17)
GATHERING_ORIENTATION_BOUNDING_BOX = {  # x_min, x_max, y_min, y_max offset
    0: [-10, 0, -8, +8],  # up
    1: [-8, +8, 0, +10],  # right
    2: [0, +10, -8, +8],  # down
    3: [-8, +8, -10, 0],  # left
}

GATHERING_ACTIONS = BASE_ACTIONS.copy()
GATHERING_ACTIONS.update({
    7: 'use beam',
})

GATHERING_COLOR = BASE_GRID_COLOR.copy()
GATHERING_COLOR.update({
    'U': [0, 0, 255],  # blue player
    'V': [255, 0, 0],  # red player
    'W': [102, 0, 204],  # purple, overlapp, also contains apple and orientation
    'O': [255, 140, 0],  # Deep orange, for the opponent
    'S': [0, 255, 255],  # light blue for self
    'P': [0, 255, 0],  # green for apples
    'F': [255, 255, 204], # light yellow for beam&player
    'B': [255, 255, 0], # yellow for beam
    'C': [100, 30, 30], # brown for collect apple
})
GATHERING_COLOR.update({
    '_': [47, 79, 79],  # DarkSlateGray
    '&': [169, 169, 169],  # Dark grey
    'u': [0, 170, 255],  # light blue (dark than s)
    'v': [255, 0, 162],  # pink
    'w': [130, 70, 190],  # light purple
    's': [0, 255, 255],  # Light Blue
    'o': [255, 200, 130],  # Light Orange
    'p': [0, 255, 0],  # Apple
    'f': [255, 255, 204],
    'b': [255, 255, 0], # yellow beam
    'c': [100, 30, 30], # apple and player
})


## USER CONFIGURATIONS
GATHERING_FPS = 5
GATHERING_RESOLUTION = [1200, 700]  # [width, hight]
GATHERING_MAP_SIZE = (30,20)
GATHERING_PLAYER_BLOOD = 2
GATHERING_TAGGED_TIME = 2
GATHERING_APPLE_NUMBER = 3
GATHERING_APPLE_RESPAWN = 3
## 30*20 map
GATHERING_MAP = [
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',
    '@@@@@                   @@@@@@@@@@@',
    '@@                               @@',
    '@@           @@                  @@',
    '@@                               @@',
    '@@                   @           @@',
    '@@                 @@            @@',
    '@@    @@@     @                  @@',
    '@@    @@@                        @@',
    '@@    @@@                        @@',
    '@@                               @@',
    '@@    @@@@@                      @@',
    '@@                               @@',
    '@@         @@@@                  @@',
    '@@                               @@',
    '@@                   @@@         @@',
    '@@   @@@@@@                      @@',
    '@@                      @@       @@',
    '@@                               @@',
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',
]
''' 50*30 map
GATHERING_MAP = [
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',
    '@@@@@@@                            @@@@@@@@@@@@@@@',
    '@@                        @@@@                 @@@',
    '@@                        @@@@                 @@@',
    '@@                        @@@@                 @@@',
    '@@                        @@@@                 @@@',
    '@@                        @@@@                 @@@',
    '@@                        @@@@                 @@@',
    '@                                        @@@@@@@@@',
    '@@@@@                           @@@@@@@@@@@@@@@@@@',
    '@                               @@@@@@@@@@@@@@@@@@',
    '@@                                        @@@@@@@@',
    '@@           @@                                 @@',
    '@@                                              @@',
    '@@                   @                          @@',
    '@@                      @@                      @@',
    '@@    @@@                    @                  @@',
    '@@    @@@                        @@@@           @@',
    '@@    @@@                        @@@@           @@',
    '@@                                              @@',
    '@@                   @@@@@                      @@',
    '@@                      @@                      @@',
    '@@                        @@@@                  @@',
    '@@                                              @@',
    '@@                                              @@',
    '@@                   @@@@@@                     @@',
    '@@                      @@                      @@',
    '@@                           @                  @@',
    '@@                                              @@',
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',
]
'''