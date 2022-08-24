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

__all__ = [
    "ModelParams"
]

from dataclasses import dataclass


@dataclass()
class ModelParams:
    id: str = None
    node_address: str = None
    step: int = 0
    path: str = None
    loss: float = 0.0
    metric_name: str = None
    metric_value: float = 0.0
    score: float = 0.0
    is_aggregate: bool = False
