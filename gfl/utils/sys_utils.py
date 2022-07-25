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

import psutil
import pynvml
import os


_NVML_INITIALIZED = False


try:
    pynvml.nvmlInit()
    _NVML_INITIALIZED = True
except:
    _NVML_INITIALIZED = False


class SysUtils(object):
    """
    Some methods of query the usage system hardware resources.
    The following is meaning of some identifier names:

    proc_*: query resources of specified process
    pid: process id, None means current process
    index: serial number of cpu or gpu, start from 0
    """

    @classmethod
    def cpu_count(cls, logical=True):
        return psutil.cpu_count(logical)

    @classmethod
    def cpu_percent(cls, index=None):
        """
        return the average used percent if index is None or less than 0
        """
        if index is None or index < 0:
            return psutil.cpu_percent()
        else:
            return psutil.cpu_percent(percpu=True)[index]

    @classmethod
    def mem_total(cls):
        mem = psutil.virtual_memory()
        return mem.total

    @classmethod
    def mem_used(cls):
        mem = psutil.virtual_memory()
        return mem.used

    @classmethod
    def mem_available(cls):
        mem = psutil.virtual_memory()
        return mem.available

    @classmethod
    def mem_free(cls):
        mem = psutil.virtual_memory()
        return mem.free

    @classmethod
    def gpu_count(cls):
        if not _NVML_INITIALIZED:
            return 0
        return pynvml.nvmlDeviceGetCount()

    @classmethod
    def gpu_mem_total(cls, index):
        if not _NVML_INITIALIZED:
            return 0
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return mem.total

    @classmethod
    def gpu_mem_used(cls, index):
        if not _NVML_INITIALIZED:
            return 0
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return mem.used

    @classmethod
    def gpu_mem_free(cls, index):
        if not _NVML_INITIALIZED:
            return 0
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return mem.free

    @classmethod
    def gpu_utilization_rates(cls, index):
        if not _NVML_INITIALIZED:
            return 0
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        rate = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return rate

    @classmethod
    def proc_cpu_percent(cls, pid=None):
        if pid is None:
            id1=os.getpid()
            p = psutil.Process(id1)
        else:
            p = psutil.Process(pid)
        return p.cpu_percent(interval=0.05)  # TODO: not works

    @classmethod
    def proc_mem_used(cls, pid=None):
        if pid is None:
            id1 = os.getpid()
            p = psutil.Process(id1)
        else:
            p = psutil.Process(pid)
        return p.memory_info().rss   # TODO: mem_used(B), not memory percent

    @classmethod
    def proc_gpu_mem_used(cls, index, pid=None):
        if not _NVML_INITIALIZED:
            return 0
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        info_list = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        info_list_len = len(info_list)
        gpu_memory_used = 0
        if info_list_len > 0:
            for i in info_list:
                if i.pid == pid:
                    gpu_memory_used += i.usedgpumemory
        return gpu_memory_used
