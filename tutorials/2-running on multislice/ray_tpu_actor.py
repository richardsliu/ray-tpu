import ray
import ray_tpu
import logging


logging.basicConfig(level=logging.DEBUG,  # Set the desired logging level
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


@ray_tpu.remote(
    topology={"v4-16": 2},
    multislice=True,
)
class MyActor:
    def __init__(self, data: str):
        self._data = data

    def my_task(self):
        return self._data

ray.init()

a = MyActor(data="hello actor")
print(ray.get(a.my_task()))
