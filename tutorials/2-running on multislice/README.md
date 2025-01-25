# Running on Multislice

In this tutorial we'll look at how to run a workload on multislice TPUs. For an
introduction to multislice training, check out [this
link](https://cloud.google.com/tpu/docs/multislice-introduction).

## Deploying a Multislice Cluster

Modify the `cluster.yaml` file by modifying the maximum number of workers:

```bash
cluster_name: cluster # MODIFY choose your cluster name
max_workers: 2
```

Then modify the number of workers in the TPU section:

```bash
  # The ray worker nodes are TPU nodes
  ray_tpu:
    min_workers: 2
    max_workers: 2
```

Then use `ray up cluster.yaml` to bring up the cluster.

## Running a Multislice Workload

We'll take the Ray task from the previous tutorial and modify it slightly, by
running the remote function on 2 `v4-16` slices. We'll also set the `multislice`
flag to `True`:


```python
import ray
import ray_tpu


@ray_tpu.remote(
    topology={"v4-16": 2},
    multislice=True,
)
def my_task():
    return "hello world"

ray.init()

print(ray.get(my_task()))
```

Setting the `multislice` flag injects environment variables that enables the
slices to coordinate with each other.

Running this code should produce:

```bash
['hello world', 'hello world', 'hello world', 'hello world']
```


We can modify the actor in a similar way as well:


```python
import ray
import ray_tpu


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
```

Running this code should produce the following:

```bash
['hello actor', 'hello actor']
```
