# Running a Simple Task

After the last tutorial you should already have a Ray cluster with TPUs up and
running. Now let's try to use the cluster for some simple tasks.


## Ray Task

Let's start with a simple Ray task:

```python
import ray
import ray_tpu


@ray_tpu.remote(
    topology={"v4-16": 1},
)
def my_task():
    return "hello world"

ray.init()

print(ray.get(my_task()))
```

The `ray_tpu.remote` decorator should look very familiar -- it's nearly
identical to how a typical ray decorator looks. The `topology` map describes
that the function `my_task` should be scheduled on a single `v4-16` TPU slice.

Running this code should produce:

```bash
['hello world', 'hello world']
```

The output `hello world` is printed twice because it is returned from each of
the two TPU hosts.


## Ray Actor

Now let's see if the same works for Ray actors.

```python
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

a = MyActor(data="hello actor")
print(ray.get(a.my_task()))
```

This is similar to our first example, except we are now decorating a class
instead of a function. The semantics remain the same - the actor should be
scheduled on a single v4-16 slice.

Running this code should produce the following:

```bash
['hello actor', 'hello actor']
```

In the following tutorial, we'll see how to leverage `ray-tpu` to run on
multiple TPU slices.
