# Getting Started

This tutorial will walk through a very basic setup of a multi-host TPU.

## Deploying a Ray Cluster

An example `cluster.yaml` manifest file has been provided for you. This
configuration will deploy a v4-16 TPU with 2 hosts in the us-central2-b zone.
Please modify this file according to your own GCP settings and ensure that your
project has sufficient quota and capacity.

Start the Ray Cluster with the following command:

```bash
ray up cluster.yaml
```


You can ssh into the head node and check the Ray cluster status by running the
following commands:

```bash
ray attach cluster.yaml
ray status
```


This should show something like:

```bash
======== Autoscaler status: 2025-01-18 01:35:02.400986 ========
Node status
---------------------------------------------------------------
Active:
 1 ray_head_default
 1 ray_tpu
Pending:
 (no pending nodes)
Recent failures:
 (no failures)

Resources
---------------------------------------------------------------
Usage:
 0.0/480.0 CPU
 0.0/8.0 TPU
 0.0/1.0 TPU-v4-16-head
 0B/591.46GiB memory
 0B/255.96GiB object_store_memory
 0.0/2.0 ray-ricliu-v4-16-worker-b3e869d0-tpu

Demands:
 (no resource demands)
```

Note that the Ray cluster has 8 TPUs as expected. There are also other custom
resources provided by the machines in the cluster.

Now let's use the `ray-tpu` library to detect the TPU topology. Start a python
session on the Ray head:

```bash
$ python
Python 3.10.12 (main, Nov  6 2024, 20:22:13) [GCC 11.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import ray_tpu
>>> ray_tpu.init()
2025-01-18 01:45:23,388 INFO worker.py:1636 -- Connecting to existing Ray cluster at address: 10.130.0.139:6379...
2025-01-18 01:45:23,400 INFO worker.py:1812 -- Connected to Ray cluster. View the dashboard at 10.130.0.139:8265
>>> ray_tpu.available_resources()
{'v4-16': [RayTpu(name='ray-ricliu-v4-16-worker-b3e869d0-tpu', num_hosts=2, chips_per_host=4, head_ip='10.130.0.10', topology='v4-16')]}
```

The library should return the detected topology (`v4-16`), along with other
metadata:
  * num_hosts: the total number of TPU hosts in this topology (2)
  * chips_per_host: the number of TPU chips on each host
  * head_ip: the IP of the head worker (we'll come back to this in later
    tutorials)


In the next tutorial, we'll look at how to run a Ray task or actor using
familiar Ray syntax.


