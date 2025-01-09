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
import logging
from typing import Any, List, Mapping, Optional, Type, Union
import socket
import ray
from dataclasses import dataclass
import time


TPU_HEAD_PATTERN = r"TPU-(.+)-head"


@dataclass
class RayTpu:
  name: str
  num_hosts: int
  chips_per_host: int
  head_ip: str
  topology: str


class RayTpuManager:
  #@classmethod
  #def get_available_resources(cls) -> Mapping[str, RayTpu]:

  def initialize(self):
    tpu_pattern = re.compile(TPU_HEAD_PATTERN)
    self.resources = {}

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

    available_resources = ray.available_resources()
    logging.info("Ray available resources: %s", available_resources)
    for key, value in available_resources.items():
      match = tpu_pattern.match(key)
      if match:
        topology = f"{match.group(1)}"
        topology_key = key
        num_tpu_pods = int(value)
        logging.info("Found %d TPU pods of type: %s", num_tpu_pods, topology)
        metadata_handles = []
        for _ in range(num_tpu_pods):
          metadata_handles.append(_get_tpu_pod_metadata.options(resources={topology_key: 1}).remote())
        logging.debug("Gathering TPU pod metadata")
        metadata = ray.get(metadata_handles)

        self.resources[topology] = []
        for tpu_name, num_hosts, chips_per_host, head_ip in metadata:
          self.resources[topology].append(
              RayTpu(
                  name=tpu_name,
                  num_hosts=num_hosts,
                  chips_per_host=chips_per_host,
                  head_ip=head_ip,
                  topology=topology,
              )
          )


  def get_available_resources(self) -> Mapping[str, RayTpu]:
    return self.resources


  def remote(
      self,
      actor_or_fn: Union[ray.actor.ActorClass, Type],
      topology: Optional[Mapping[str, int]] = None,
      multislice,
      env: Optional[Mapping[str, Any]] = None,
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

    Returns:
        A list of ActorHandles or ObjectRefs.
    """
    if env is None:
      env = {}

    if isinstance(actor_or_fn, type):
      actor_or_fn = ray.remote(actor_or_fn)
    elif callable(actor_or_fn):
      if not hasattr(actor_or_fn, "remote"):
        actor_or_fn = ray.remote(actor_or_fn)
    elif not isinstance(actor_or_fn, ray.actor.ActorClass):
      raise AssertionError(f"`actor_or_fn` should be a class definition, ActorClass, or callable, got {type(actor_or_fn)}")

    handles = []

    if len(topology) > 1:
      raise AssertionError("Only single topology types are supported")

    tpu_id, count = topology.popitem()

    if not tpu_id in self.resources:
      raise AssertionError(f"{tpu_id} is not a known topology type")

    tpu = self.resources[tpu_id]


    if multislice:
      logging.info("Scheduling with multislice.")
      coordinator_port = 8081
      mxla_env = {
          "MEGASCALE_COORDINATOR_ADDRESS": f"{tpus[0].head_ip}:{coordinator_port}",
          "MEGASCALE_NUM_SLICES": str(len(tpus)),
          "MEGASCALE_PORT": f"{coordinator_port}",
          "MEGASCALE_SLICE_ID": str(tpu_id),
      }
      env_vars = env | mxla_env
      logging.debug("Env vars being set: %s", env_vars)
      # Schedule on the lead worker first to consume the HEAD resource
      handles += [
          actor_or_fn.options(
              runtime_env={"env_vars": env_vars}, resources={"TPU": tpu.chips_per_host, tpu.name: 1, f"TPU-{tpu.topology}-head": 1}
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
          actor_or_fn.options(resources={"TPU": tpu.chips_per_host, tpu.name: 1, f"TPU-{tpu.topology}-head": 1}).remote(*args, **kwargs)
      ]
      time.sleep(1)
      handles += [
          actor_or_fn.options(resources={"TPU": tpu.chips_per_host, tpu.name: 1}).remote(*args, **kwargs) for _ in range(tpu.num_hosts - 1)
      ]
    return handles
