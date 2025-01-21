import ray
import ray_tpu
from train import main as maxtext_main

import logging
from typing import Sequence
from absl import app


# Default env vars that run on all TPU VMs.
MACHINE_ENV_VARS = {
    "ENABLE_PJRT_COMPATIBILITY": "true",
    "TPU_SLICE_BUILDER_DUMP_CHIP_FORCE": "true",
    "TPU_SLICE_BUILDER_DUMP_ICI": "true",
    "XLA_FLAGS": "--xla_dump_to=/tmp/xla_dump_file --xla_dump_hlo_as_proto",  # Dumps HLOs for debugging
}


def setup_loggers():
  """Sets up loggers for Ray."""
  logging.basicConfig(level=logging.INFO)


@ray_tpu.remote(
    topology={"v4-16": 1},
)
def run_maxtext_train(argv: Sequence[str]):
    maxtext_main(argv=argv)


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


if __name__ == "__main__":
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)
  app.run(main)
