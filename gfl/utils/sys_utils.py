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
        pass

    @classmethod
    def gpu_mem_total(cls, index):
        pass

    @classmethod
    def gpu_mem_used(cls, index):
        pass

    @classmethod
    def gpu_mem_free(cls, index):
        pass

    @classmethod
    def gpu_utilization_rates(cls, index):
        pass

    @classmethod
    def proc_cpu_percent(cls, pid=None):
        pass

    @classmethod
    def proc_mem_used(cls, pid=None):
        pass

    @classmethod
    def proc_gpu_mem_used(cls, pid=None, index=None):
        pass

