import ray
import ray_tpu
import logging


logging.basicConfig(level=logging.DEBUG,  # Set the desired logging level
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


@ray_tpu.remote(
    topology={"v4-16": 1},
)
def my_task():
    return "hello world"

ray.init()

print(ray.get(my_task()))
