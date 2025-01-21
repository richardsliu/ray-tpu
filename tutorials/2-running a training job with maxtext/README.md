# Running a Training Job with MaxText

In this tutorial we'll run a real training job with Ray using TPUs. The framework
we'll use is
[MaxText](https://github.com/AI-Hypercomputer/maxtext), a scalable and high
performance open source library for training LLMs using JAX and XLA.


## Installation

We'll modify our Ray cluster yaml slightly, to install MaxText and its
dependencies on our TPU VMs. Add the following lines to the setup steps:

```bash
  - git clone https://github.com/AI-Hypercomputer/maxtext
  - pip install -r maxtext/requirements.txt
```

Bring up the Ray cluster and attach a session:

```bash
ray up cluster.yaml
ray attach cluster.yaml
```


## Adding a Ray Trainer

MaxText contains a training script
[train.py](https://github.com/AI-Hypercomputer/maxtext/blob/main/MaxText/train.py) which needs to run
on each TPU host in a topology. This is similar to other SPMD (single program,
multiple data) ML workloads. We can achieve this easily with `ray-tpu` by
creating a wrapper around the `trainer.py` main function:


```python
@ray_tpu.remote(
    topology={"v4-16": 1},
)
def run_maxtext_train(argv: Sequence[str]):
    maxtext_main(argv=argv)
```


Next step is simply invoking this function call in our own main function:

```python
def main(argv: Sequence[str]):
  ray.init(runtime_env=dict(worker_process_setup_hook=setup_loggers))
  ray_tpu.init()

  logging.info(f"argv: {argv}")

  try:
    ray.get(run_maxtext_train(argv=argv))
  except Exception as e:
    logging.error("Caught error during training: %s", e)
    logging.error("Shutting down...")
    ray.shutdown()
    raise e

  logging.info("Training complete!")
  ray.shutdown()
```

Save the changes in a file called `maxtext_ray_trainer.py`.

We can start training with the following command:

```bash
    python MaxText/ray_trainer.py MaxText/configs/base.yml \
        base_output_directory=/tmp/maxtext \
        dataset_type=synthetic \
        per_device_batch_size=2 \
        max_target_length=8192 \
        model_name=default \
        steps=100 \
        run_name=test
```


You should see some output on the console that shows the completed training
steps:



```bash
(run_maxtext_train pid=78967, ip=10.130.0.11) Started an asynchronous checkpoint save for step 0
(run_maxtext_train pid=78967, ip=10.130.0.11)
(run_maxtext_train pid=78967, ip=10.130.0.11) Memstats: After params initialized:
(run_maxtext_train pid=78967, ip=10.130.0.11)   Using (GB) 1.59 / 30.75 (5.170732%) on TPU_4(process=1,(0,0,1,0))
(run_maxtext_train pid=78967, ip=10.130.0.11)   Using (GB) 1.59 / 30.75 (5.170732%) on TPU_5(process=1,(1,0,1,0))
(run_maxtext_train pid=78967, ip=10.130.0.11)   Using (GB) 1.59 / 30.75 (5.170732%) on TPU_6(process=1,(0,1,1,0))
(run_maxtext_train pid=78967, ip=10.130.0.11)   Using (GB) 1.59 / 30.75 (5.170732%) on TPU_7(process=1,(1,1,1,0))
(run_maxtext_train pid=78967, ip=10.130.0.11) completed step: 0, seconds: 11.775, TFLOP/s/device: 13.153, Tokens/s/device: 1391.395, total_weights: 131072, loss: 12.066
(run_maxtext_train pid=80538, ip=10.130.0.12)
(run_maxtext_train pid=80538, ip=10.130.0.12) To see full metrics 'tensorboard --logdir=/tmp/maxtext/test/tensorboard/'
(run_maxtext_train pid=80538, ip=10.130.0.12) Waiting for step 0 to finish before checkpoint...
(run_maxtext_train pid=80538, ip=10.130.0.12) Waited 0.7087039947509766 seconds for step 0 to finish before starting checkpointing.
(run_maxtext_train pid=80538, ip=10.130.0.12) Started an asynchronous checkpoint save for step 0
(run_maxtext_train pid=80538, ip=10.130.0.12) Memstats: After params initialized:
(run_maxtext_train pid=80538, ip=10.130.0.12)   Using (GB) 1.59 / 30.75 (5.170732%) on TPU_3(process=0,(1,1,0,0)) [repeated 4x across cluster]
(run_maxtext_train pid=78967, ip=10.130.0.11) completed step: 4, seconds: 1.116, TFLOP/s/device: 138.799, Tokens/s/device: 14683.240, total_weights: 131072, loss: 0.000 [repeated 9x across cluster]
(run_maxtext_train pid=80538, ip=10.130.0.12) completed step: 9, seconds: 1.068, TFLOP/s/device: 145.065, Tokens/s/device: 15346.083, total_weights: 131072, loss: 0.000 [repeated 9x across cluster]
(run_maxtext_train pid=78967, ip=10.130.0.11) completed step: 14, seconds: 1.116, TFLOP/s/device: 138.754, Tokens/s/device: 14678.439, total_weights: 131072, loss: 0.000 [repeated 10x across cluster]

...

(run_maxtext_train pid=78967, ip=10.130.0.11) completed step: 89, seconds: 1.116, TFLOP/s/device: 138.760, Tokens/s/device: 14679.083, total_weights: 131072, loss: 0.000 [repeated 10x across cluster]
(run_maxtext_train pid=80538, ip=10.130.0.12) completed step: 94, seconds: 1.091, TFLOP/s/device: 141.924, Tokens/s/device: 15013.837, total_weights: 131072, loss: 0.000 [repeated 10x across cluster]
(run_maxtext_train pid=78967, ip=10.130.0.11) completed step: 99, seconds: 1.116, TFLOP/s/device: 138.763, Tokens/s/device: 14679.412, total_weights: 131072, loss: 0.000 [repeated 10x across cluster]
(run_maxtext_train pid=80538, ip=10.130.0.12) Output size: 1657041920, temp size: 4907988480, argument size: 1657366016, host temp size: 0, in bytes.
I0121 01:39:46.830807 130655182204928 ray_trainer.py:47] Training complete!
(run_maxtext_train pid=80538, ip=10.130.0.12) completed step: 99, seconds: 1.191, TFLOP/s/device: 130.014, Tokens/s/device: 13753.874, total_weights: 131072, loss: 0.000
```


