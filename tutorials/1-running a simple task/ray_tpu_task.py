import ray
import ray_tpu


@ray_tpu.remote(
    topology={"v4-16": 1},
)
def my_task():
    return "hello world"

ray.init()

ray_tpu.init()


print(ray_tpu.available_resources())

print(ray.get(my_task()))
