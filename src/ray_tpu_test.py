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
