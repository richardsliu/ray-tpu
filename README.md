# Ray TPU

This is a repository containing utilities for running [Ray](https://www.ray.io/)
on Cloud TPUs. For more information about TPUs, please check out the official
Google Cloud documentation [here](https://cloud.google.com/tpu).

## Why this package?

TPUs are different from other accelerators like GPUs because they are
"pod-centric". Scheduling jobs and workloads on TPUs require awareness of slice
topologies and other factors. This package introduces higher level utilities
that simplify running Ray workloads on TPU pod slices as if they were single
nodes.

## Installation

Run the following command to install the package:
```
pip install ray-tpu
```

## Usage Examples


Example Ray task:
```
import ray
import logging
import ray_tpu


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

ray.init()
ray_tpu.init()

print("Cluster resources: ", ray_tpu.cluster_resources())
print("Available resources: ", ray_tpu.available_resources())


@ray_tpu.remote(
    accelerator_type="v4-128",
    num_slices=2,
    with_mxla=True,
    env={
        "TPU_STDERR_LOG_LEVEL": "0", "TPU_MIN_LOG_LEVEL": "0", "TF_CPP_MIN_LOG_LEVEL": "0"
    },
)
def test():
    import jax
    return jax.device_count()

print("Running test")
print(ray.get(test()))

ray.shutdown()
```

Example Ray actor:

```
import ray
import logging
import ray_tpu


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

ray.init()
ray_tpu.init()

print("Cluster resources: ", ray_tpu.cluster_resources())
print("Available resources: ", ray_tpu.available_resources())


@ray_tpu.remote(
    accelerator_type="v4-128",
    num_slices=2,
    with_mxla=True
)
class Test:
    def __init__(self, a: str):
        self._a = a

    def print(self):
        print(self._a)

    def test(self):
        import jax
        return jax.device_count()


a = Test()
ray.get(a.test())
```

