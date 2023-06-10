import os

import ray

if __name__ == "__main__":
    ray.init(address='auto', _redis_password=os.environ['redis_password'])
