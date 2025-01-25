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

Note that the Ray cluster has 8 TPUs as expected.

In the next tutorial, we'll look at how to run a Ray task or actor using
familiar Ray syntax.


