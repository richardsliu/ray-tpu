import os
import ray

import torch_xla.runtime as xr
from torch_xla._internal import pjrt

@ray.remote
def init_multiprocess(local_world_size: int):
  local_rank = int(os.environ['TPU_VISIBLE_CHIPS'])

  pjrt.initialize_multiprocess(local_rank, local_world_size)
  xr._init_world_size_ordinal()
