# Basic Tutorial

This tutorial will walk through a very basic example using a multi-host TPU.

This example will use a v6e-8 TPU.

## Setup

Run the following gcloud command:

```bash
    gcloud compute tpus tpu-vm create $TPU_NAME \
        --zone=$ZONE \
        --accelerator-type=v6e-8 \
        --version=v2-alpha-tpuv6e
```

After the TPU instance is deployed, ssh into the VM:

```bash
    gcloud compute ssh $TPU_NAME --zone=$ZONE
```


## Running a Simple Workload

After logging onto the TPU VM, we can try to run a simple workload.

First, install the dependent packages:

```bash
pip install ray
pip install ray_tpu
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

Now let's start a Python shell. Input the following function:

```python

def jax_test()
  import jax
  return jax.device_count()
```

This function simply returns the number of JAX devices found on the local
machine. Since we are running on a 2-host TPU machine with 4 chips on each,
this should return `[4, 4]`.

Now we'll start by initializing Ray:

```python
import ray
ray.init()
```

You should see some output like the following, if the cluster is able to
initialize properly:

```bash
2024-12-10 01:38:00,284 INFO worker.py:1821 -- Started a local Ray instance.
RayContext(dashboard_url='', python_version='3.10.12', ray_version='2.40.0', ray_commit='22541c38dbef25286cd6d19f1c151bf4fd62f2ed')
```

Next we'll initialize the `RayTpuManager`:

```python
from ray_tpu import RayTpuManager
tpu_resources = RayTpuManager.get_available_resources()
```

If you examine the contents of `tpu_resources`, you should see the following:

```bash
{'v6e-8': [RayTpu(name='$TPU_NAME', num_hosts=2, head_ip='10.130.0.20', topology='v6e-8')]}
```

This means that we are able to identify the running TPU VM as a 2-host V6E
machine with 8 chips. We've also identified the IP of the head TPU VM.

Next we'll call our `jax_test` function with Ray:

```python
tasks = RayTpuManager.remote(tpus=tpu_resources['v6e-8'], multislice=True, actor_or_fn=test)
ray.get(tasks)
```

This should output `[4, 4]` as expected.


