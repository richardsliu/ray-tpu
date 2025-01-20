import ray
import ray_tpu


@ray_tpu.remote(
    topology={"v4-16": 1},
)
class MyActor:
    def __init__(self, data: str):
        self._data = data

    def my_task(self):
        return self._data

ray.init()

ray_tpu.init()


print(ray_tpu.available_resources())

a = MyActor(data="hello actor")
print(ray.get(a.my_task()))
