import os, sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from core.environments.utils import ascii_list_to_array, ascii_array_to_rgb_array, ascii_array_to_str, ascii_dict_to_color_array
from core.environments.gathering.constants import GATHERING_MAP, GATHERING_COLOR
from core.environments.gathering.env import Gathering
import numpy as np
from gymnasium.utils import seeding
from PIL import Image

rng, _ = seeding.np_random(114514)
a = Gathering(rng, )
a.render_mode = 'rgb_array'
a.reset()
print(ascii_array_to_str(a.grid_world()))
img = Image.fromarray(a.render()).save('sample.png')

out = a.step(dict(zip(a.all_agents, np.random.choice(7, size=(len(a.all_agents),)).tolist())))
output = a.grid_world()
# print(output)
rgb = ascii_array_to_rgb_array(output, a.ascii_color_array)
img = Image.fromarray(a.render()).save("sample1.png")

from pettingzoo.test import parallel_api_test
from pettingzoo.butterfly import cooperative_pong_v5
from core.environments.gathering.env import GatheringEnv
env = GatheringEnv()
parallel_api_test(env, num_cycles=100000000)