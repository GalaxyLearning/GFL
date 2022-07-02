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
    "JobConfig",
    "TrainConfig",
    "AggregateConfig",
    "DatasetConfig"
]

from dataclasses import dataclass


@dataclass()
class JobConfig:

    trainer: str
    aggregator: str


@dataclass()
class TrainConfig:

    model: str
    optimizer: str
    criterion: str
    lr_scheduler: str = None
    epoch: int = 10
    batch_size: int = 32


@dataclass()
class AggregateConfig:

    global_epoch: int = 50


@dataclass()
class DatasetConfig:

    dataset: str
    val_dataset: str = None
    val_rate: float = 0.2
