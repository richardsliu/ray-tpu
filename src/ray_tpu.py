# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Experimental utilities for running Ray with Cloud TPU."""

import re
import inspect
from functools import partial
import logging
from typing import Any, Callable, List, Mapping, Optional, Type, Union
import socket
import ray
from ray.util.placement_group import (
    placement_group,
    placement_group_table,
    remove_placement_group,
)
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from dataclasses import dataclass
import time


TPU_HEAD_PATTERN = r"TPU-(.+)-head"


@dataclass
class RayTpu:
  name: str
  num_hosts: int
  chips_per_host: int
  head_ip: str


class RayTpuManager:

  def fetch_metadata(self, pgs):
    @ray.remote
    def _get_tpu_pod_metadata():
      """Gets the TPU metadata from TPU leaders."""
      # avoid race conditions
      time.sleep(3)
      tpu_name = ray.util.accelerators.tpu.get_current_pod_name()
      num_hosts = ray.util.accelerators.tpu.get_current_pod_worker_count()
      # TODO: replace with ray.util.accelerators.tpu.get_num_tpu_chips_on_node
      chips_per_host = ray._private.accelerators.TPUAcceleratorManager.get_current_node_num_accelerators()
      ip = socket.gethostbyname(socket.gethostname())
      return tpu_name, num_hosts, chips_per_host, ip

    tpu_info = []
    for pg in pgs:
        metadata_handle = _get_tpu_pod_metadata.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg,)
        ).remote()
        tpu_name, num_hosts, chips_per_host, head_ip = ray.get(metadata_handle)

        tpu_info.append(
                RayTpu(
                    name=tpu_name,
                    num_hosts=num_hosts,
                    chips_per_host=chips_per_host,
                    head_ip=head_ip,
                )
        )
    return tpu_info

  def reserve(
      self,
      topology_id: str,
      count: int,
      timeout: Optional[int] = 600):
    """
    This is a hacky way to reserve the topology. It utilizes Ray's
    existing ability to autoscale based on the "TPU-X-head" syntax
    and then fetches the metadata from the placement group in order
    to schedule the subsequent tasks. The placement group itself is
    deleted once the metadata is fetched.

    The longer term plan is to replace this with the locality group
    implementation.
    """
    tpu_head = f"TPU-{topology_id}-head"
    logging.info(f"Placement groups {tpu_head} are creating...")
    pgs = []
    for i in range(count):
      pg = placement_group([{tpu_head: 1, "CPU": 1}])
      ray.get(pg.ready(), timeout=timeout)
      logging.info(f"Placement group {tpu_head} created.")
      pgs.append(pg)

    tpu_info = self.fetch_metadata(pgs)
    logging.info(f"Fetched metadata: {tpu_info}")

    for pg in pgs:
      remove_placement_group(pg)

    return tpu_info

  def _remote_host_mode(
      self,
      topology_id: str,
      actor_or_fn: Union[ray.actor.ActorClass, Type],
      tpu_info: List[Any],
      multislice = False,
      env: Optional[Mapping[str, Any]] = None,
      *args,
      **kwargs,
  ) -> List[Union[ray.actor.ActorHandle, ray._raylet.ObjectRef]]:
  
    if env is None:
      env = {}

    handles = []
    tpu_head = f"TPU-{topology_id}-head"  
    for tpu in tpu_info:
      if multislice:
        logging.info("Scheduling with multislice.")
        coordinator_port = 8081
        mxla_env = {
            "MEGASCALE_COORDINATOR_ADDRESS": f"{tpu_info[0].head_ip}:{coordinator_port}",
            "MEGASCALE_NUM_SLICES": str(len(tpu_info)),
            "MEGASCALE_PORT": f"{coordinator_port}",
            "MEGASCALE_SLICE_ID": str(i),
        }
        env_vars = env | mxla_env
        logging.debug("Env vars being set: %s", env_vars)
        # Schedule on the lead worker first to consume the HEAD resource
        handles += [
            actor_or_fn.options(
                runtime_env={"env_vars": env_vars}, resources={"TPU": tpu.chips_per_host, tpu.name: 1, tpu_head: 1}
            ).remote(*args, **kwargs)
        ]
        time.sleep(1)
        # Schedule the remaining workers.
        handles += [
            actor_or_fn.options(runtime_env={"env_vars": env_vars}, resources={"TPU": tpu.chips_per_host, tpu.name: 1}).remote(
                *args, **kwargs
            )
            for _ in range(tpu.num_hosts - 1)
        ]
      else:
        # Schedule on the lead worker first to consume the HEAD resource
        handles += [
            actor_or_fn.options(resources={"TPU": tpu.chips_per_host, tpu.name: 1, tpu_head: 1}).remote(*args, **kwargs)
        ]
        time.sleep(1)
        handles += [
            actor_or_fn.options(resources={"TPU": tpu.chips_per_host, tpu.name: 1}).remote(*args, **kwargs) for _ in range(tpu.num_hosts - 1)
        ]
    return handles

  def _remote_device_mode(
      self,
      topology_id: str, 
      actor_or_fn: Union[ray.actor.ActorClass, Type],
      tpu_info: List[Any],
      multislice = False,
      env: Optional[Mapping[str, Any]] = None,
      *args,
      **kwargs,
  ) -> List[Union[ray.actor.ActorHandle, ray._raylet.ObjectRef]]:
    from pjrt_util import init_multiprocess

    if env is None:
      env = {}

    handles = []
    tpu_head = f"TPU-{topology_id}-head"
    for tpu in tpu_info:
      device_count = tpu.chips_per_host * tpu.num_hosts

      # Init PJRT
      init_handle = init_multiprocess.options(resources={"TPU": 1, tpu.name: 1}).remote(local_world_size=tpu.chips_per_host)
      ray.get(init_handle)
        
      if multislice:
        logging.info("Scheduling with multislice.")
        coordinator_port = 8081
        mxla_env = {
            "MEGASCALE_COORDINATOR_ADDRESS": f"{tpu_info[0].head_ip}:{coordinator_port}",
            "MEGASCALE_NUM_SLICES": str(len(tpu_info)),
            "MEGASCALE_PORT": f"{coordinator_port}",
            "MEGASCALE_SLICE_ID": str(i),
        }
        env_vars = env | mxla_env
        logging.debug("Env vars being set: %s", env_vars)
        # Schedule on the lead worker first to consume the HEAD resource
        handles += [
            actor_or_fn.options(
                runtime_env={"env_vars": env_vars}, resources={"TPU": 1, tpu.name: 1, tpu_head: 1}
            ).remote(*args, **kwargs)
        ]
        time.sleep(1)
        # Schedule the remaining workers.
        handles += [
            actor_or_fn.options(runtime_env={"env_vars": env_vars}, resources={"TPU": 1, tpu.name: 1}).remote(
                *args, **kwargs
            )
            for _ in range(device_count - 1)
        ]
      else:
        # Schedule on the lead worker first to consume the HEAD resource
        handles += [
            actor_or_fn.options(resources={"TPU": 1, tpu.name: 1, tpu_head: 1}).remote(*args, **kwargs)
        ]
        time.sleep(1)
        handles += [
            actor_or_fn.options(resources={"TPU": 1, tpu.name: 1}).remote(*args, **kwargs) for _ in range(device_count - 1)
        ]
    return handles


  def remote(
      self,
      actor_or_fn: Union[ray.actor.ActorClass, Type],
      topology: Optional[Mapping[str, int]] = None,
      multislice = False,
      device_mode = False,
      env: Optional[Mapping[str, Any]] = None,
      timeout: Optional[int] = 600,
      *args,
      **kwargs,
  ) -> List[Union[ray.actor.ActorHandle, ray._raylet.ObjectRef]]:
    """Schedules an actor or function on a set of TPUs.

    Args:
        actor_or_fn: The definition of the actor, as a class or as a remote class, OR a function,
            as a function or executable remote task.
        topology: A dictionary representing the TPU topology, e.g. {"v6e-8": 1}
        multislice: Whether or not to schedule this actor with multislice technology.
            If set to true, this injects the metadata needed to schedule a multislice workload.
            Else, this will be treated as individual pod slices.
        env: An optional base environment, as a dictionary.
        timeout: Timeout in seconds for placement group creation. Defaults to 600.

    Returns:
        A list of ActorHandles or ObjectRefs.
    """
    if isinstance(actor_or_fn, type):
      actor_or_fn = ray.remote(actor_or_fn)
    elif callable(actor_or_fn):
      if not hasattr(actor_or_fn, "remote"):
        actor_or_fn = ray.remote(actor_or_fn)
    elif not isinstance(actor_or_fn, ray.actor.ActorClass):
      raise AssertionError(f"`actor_or_fn` should be a class definition, ActorClass, or callable, got {type(actor_or_fn)}")

    if len(topology) > 1:
      raise AssertionError("Only single topology types are supported")

    topology_id, count = topology.popitem()

    # topology_id is in the form "{generation}-{cores}".
    # count is how many instances of each topology to create.

    tpu_info = self.reserve(topology_id, count, timeout)
      
    time.sleep(1)

    if len(tpu_info) != count:
        raise AssertionError("Number of TPUs not equal to requested amount")

    if device_mode:
        return self._remote_device_mode(
          topology_id,
          actor_or_fn, 
          tpu_info,
          multislice,
          env,
          *args,
          **kwargs)
    else:
        return self._remote_host_mode(
          topology_id,
          actor_or_fn,
          tpu_info,
          multislice,
          env,
          *args,
          **kwargs)        


_manager = RayTpuManager()


def _remote_func_wrapper(
    f: Callable[[Any], Any],
    topology: Optional[Mapping[str, int]] = None,
    multislice = False,
    device_mode = False,
    env: Optional[Mapping[str, Any]] = None,
    *f_args, **f_kwargs):
    return _manager.remote(
        actor_or_fn=f,
        topology=topology,
        multislice=multislice,
        device_mode=device_mode,
        env=env,
        *f_args, **f_kwargs
    )


class _RemoteClassWrapper:
    def __init__(
        self,
        cls: type,
        topology: Optional[Mapping[str, int]] = None,
        multislice = False,
        device_mode = False,
        env: Optional[Mapping[str, Any]] = None):
        self.cls = cls
        self.topology = topology
        self.multislice = multislice
        self.device_mode = device_mode
        self.env = env

    def __call__(self, *args, **kwargs):
        self.instances = _manager.remote(
            actor_or_fn=self.cls,
            topology=self.topology,
            multislice=self.multislice,
            device_mode=self.device_mode,
            env=self.env,
            *args, **kwargs)
        return self

    def __getattr__(self, key):
        all_values = [
            getattr(inst, key) for inst in self.instances
        ]
        if callable(all_values[0]):
            def _wrapper(*args, **kwargs):
                return [
                    func.remote(*args, **kwargs) for func in all_values
                ]
            return _wrapper
        return all_values


def remote(
    topology: Optional[Mapping[str, int]] = None,
    multislice = False,
    device_mode = False,
    env: Optional[Mapping[str, Any]] = None,
):
    def decorator(f_or_c: Union[Callable[Any, Any], type]):
        if inspect.isfunction(f_or_c):
            return partial(
                _remote_func_wrapper,
                f=f_or_c,
                topology=topology,
                multislice=multislice,
                device_mode=device_mode,
                env=env)
        elif inspect.isclass(f_or_c):
            return _RemoteClassWrapper(
                f_or_c,
                topology=topology,
                multislice=multislice,
                device_mode=device_mode,
                env=env)
        else:
            raise ValueError(
                "Expected input to `ray_tpu.remote` to be a function or a class."
            )
    return decorator
