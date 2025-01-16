"""TODO: ricliu - DO NOT SUBMIT without either providing a detailed docstring or
removing it altogether.
"""


from unittest import mock
import pytest
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

@ray_tpu.remote(
    topology={"v4-16": 1},
)
def my_task():
    return "hello world"


def test_get_available_resources():
    ray_tpu.init()
    tpu_resources = ray_tpu.available_resources()


def test_ray_task():
    ray_tpu.init()
    ray.get(my_task())


def test_ray_actor():
    ray_tpu.init()
    a = MyActor(data="hello from actor")
    ray.get(a.my_task())
