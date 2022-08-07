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
        try:
            return pynvml.nvmlDeviceGetCount()
        except:
            return 0

    @classmethod
    def gpu_mem_total(cls, index):
        mem_info = cls.__gpu_memory_info(index)
        return mem_info.total if mem_info is not None else 0

    @classmethod
    def gpu_mem_used(cls, index):
        mem_info = cls.__gpu_memory_info(index)
        return mem_info.used if mem_info is not None else 0

    @classmethod
    def gpu_mem_free(cls, index):
        mem_info = cls.__gpu_memory_info(index)
        return mem_info.free if mem_info is not None else 0

    @classmethod
    def gpu_utilization_rate(cls, index):
        utilization = cls.__gpu_utilization(index)
        return utilization.gpu / 100 if utilization is not None else 0

    @classmethod
    def proc_cpu_percent(cls, pid=None):
        pid = pid or os.getpid()
        return psutil.Process(pid).cpu_percent(interval=0.05)

    @classmethod
    def proc_mem_used(cls, pid=None):
        pid = pid or os.getpid()
        return psutil.Process(pid).memory_info().rss

    @classmethod
    def proc_gpu_mem_used(cls, index, pid=None):
        pid = pid or os.getpid()
        proc = cls.__gpu_process(index, pid)
        return proc.usedGpuMemory if proc is not None else 0

    @classmethod
    def __gpu_memory_info(cls, index):
        try:
            return pynvml.nvmlDeviceGetMemoryInfo(cls.__gpu_handle(index))
        except:
            return None

    @classmethod
    def __gpu_utilization(cls, index):
        try:
            return pynvml.nvmlDeviceGetUtilizationRates(cls.__gpu_handle(index))
        except:
            return None

    @classmethod
    def __gpu_process(cls, index, pid):
        try:
            processes = pynvml.nvmlDeviceGetComputeRunningProcesses(cls.__gpu_handle(index))
            for p in processes:
                if p.pid == pid:
                    return p
            return None
        except:
            return None

    @classmethod
    def __gpu_handle(cls, index):
        if not _NVML_INITIALIZED:
            raise pynvml.NVMLError(pynvml.NVML_ERROR_UNINITIALIZED)
        return pynvml.nvmlDeviceGetHandleByIndex(index)
