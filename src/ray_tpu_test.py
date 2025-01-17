"""
Test needs to run on a Ray cluster with a v4-16.

Should output something like:
{'v4-16': [RayTpu(name='ricliu-v4-16', num_hosts=2, chips_per_host=4, head_ip='10.130.0.76', topology='v4-16')]}
['hello world', 'hello world']
['hello actor', 'hello actor']
"""
import ray
import ray_tpu
import logging

logging.basicConfig(level=logging.DEBUG,  # Set the desired logging level
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


@ray_tpu.remote(
    topology={"v4-16": 1},
)
class MyActor:
    def __init__(self, data: str):
        self._data = data

    def my_task(self):
        return self._data

@ray_tpu.remote(
    topology={"v4-16": 1},
)
def my_task():
    return "hello world"

ray.init()

ray_tpu.init()


print(ray_tpu.available_resources())

print(ray.get(my_task()))

a = MyActor(data="hello actor")
print(ray.get(a.my_task()))
