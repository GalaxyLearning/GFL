#   Copyright 2020 The GFL Authors. All Rights Reserved.
#   #
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#   #
#       http://www.apache.org/licenses/LICENSE-2.0
#   #
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

__all__ = [
    "GPUResource",
    "ComputingResource"
]


import typing
from dataclasses import dataclass, field


# the unit of below memory field is Byte

@dataclass()
class GPUResource:

    gpu_mem_used: int = 0
    gpu_mem_total: int = 0
    gpu_utilization: int = 0    # [0, 100], means percentage


@dataclass()
class ComputingResource:

    running_job_number: int = 0
    cpu_utilization: int = 0    # [0, 100], means percentage
    cpu_cores: int = 0          # the logical cpu cores
    mem_used: int = 0
    mem_total: int = 0
    gpu_number: int = 0
    gpu_mem_used: int = 0
    gpu_mem_total: int = 0
    gpu_utilization: int = 0    # [0, 100], means percentage
