#  Copyright 2020 The GFL Authors. All Rights Reserved.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import copy
from threading import Lock

import zcommons.time as time

from gfl.data.computing_resource import ComputingResource


class _Resource(object):

    def __init__(self, **kwargs):
        super(_Resource, self).__init__()
        self._resource = ComputingResource(**kwargs)
        self._update_timestamp = time.time_ms()

    @property
    def running_job_number(self):
        return self._resource.running_job_number

    @property
    def cpu_utilization(self):
        return self._resource.cpu_utilization

    @property
    def cpu_cores(self):
        return self._resource.cpu_cores

    @property
    def memory_used(self):
        return self._resource.mem_used

    @property
    def memory_total(self):
        return self._resource.mem_total

    @property
    def gpu_number(self):
        return self._resource.gpu_number

    @property
    def gpu_memory_used(self):
        return self._resource.gpu_mem_used

    @property
    def gpu_memory_total(self):
        return self._resource.gpu_mem_total

    @property
    def gpu_utilization(self):
        return self._resource.gpu_utilization

    @property
    def update_timestamp(self):
        return self._update_timestamp

    def update(self, resource: ComputingResource):
        self._resource = copy.deepcopy(resource)
        self._update_timestamp = time.time_ms()

    def add(self, resource: '_Resource'):
        self._resource.running_job_number += resource.running_job_number
        used_cpu_cores = (self.cpu_utilization * self.cpu_cores + resource.cpu_utilization * resource.cpu_cores) / 100
        self._resource.cpu_utilization = int(100 * used_cpu_cores / (self.cpu_cores + resource.cpu_cores))
        self._resource.cpu_cores += resource.cpu_cores
        self._resource.mem_used += resource.memory_used
        self._resource.mem_total += resource.memory_total
        self._resource.gpu_number += resource.gpu_number
        used_gpu = (self.gpu_utilization * self.gpu_number + resource.gpu_utilization * resource.gpu_number) / 100
        self._resource.gpu_utilization = int(100 * used_gpu / (self.gpu_number + resource.gpu_number))
        self._resource.gpu_number += resource.gpu_number
        self._resource.gpu_mem_used += resource.gpu_memory_used
        self._resource.gpu_mem_total += resource.gpu_memory_total

    @property
    def computing_resource(self):
        return copy.deepcopy(self._resource)


class ResourceManager(object):

    def __init__(self):
        super(ResourceManager, self).__init__()
        self.__resources = dict()
        self.__resources_lock = Lock()

    def update_resource(self, node_address: str, computing_resource: ComputingResource):
        with self.__resources_lock:
            if node_address not in self.__resources:
                self.__resources[node_address] = _Resource()
            resource = self.__resources[node_address]
            resource.update(computing_resource)

    def get_resource(self, node_address: str) -> ComputingResource:
        resource: _Resource = self.__resources.get(node_address, None)
        return resource.computing_resource if resource is not None else None

    def get_net_resource(self) -> ComputingResource:
        resource = _Resource()
        for _, res in self.__resources.items():
            resource.add(res)
        return resource.computing_resource if resource is not None else None
