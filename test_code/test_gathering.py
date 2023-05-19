from builtins import breakpoint
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

def rgb_onestep(env, action, step_number):
    env.step(dict(zip(env.agents, action)))
    output = env.grid_world()
    rgb = ascii_array_to_rgb_array(output, env.ascii_color_array)
    img = Image.fromarray(env.render()).save(f"gathering_small_map1/sample{step_number}.png")
    return env

## seed 1145 for sample imgs in gathering_imgs
rng, _ = seeding.np_random(189)
a = Gathering(rng, )
a.render_mode = 'rgb_array'
a.reset()
#print(ascii_array_to_str(a.grid_world()))
img = Image.fromarray(a.render()).save('gathering_small_map1/sample.png')

# np.random.choice(7, size=(len(a.agents),)).tolist()
a = rgb_onestep(a, (2,5), 1)
a = rgb_onestep(a, (5,0), 2)
a = rgb_onestep(a, (0,0), 3)
a = rgb_onestep(a, (7,0), 4)
a = rgb_onestep(a, (0,0), 5)
a = rgb_onestep(a, (0,0), 6)
a = rgb_onestep(a, (0,0), 7)
a = rgb_onestep(a, (7,0), 8)
a = rgb_onestep(a, (2,4), 9)
a = rgb_onestep(a, (0,4), 10)
a = rgb_onestep(a, (2,1), 11)
a = rgb_onestep(a, (0,3), 12)
a = rgb_onestep(a, (0,3), 13)
a = rgb_onestep(a, (2,4), 14)


from pettingzoo.test import parallel_api_test
from pettingzoo.butterfly import cooperative_pong_v5
from core.environments.gathering.env import GatheringEnv
env = GatheringEnv()
parallel_api_test(env, num_cycles=10000)