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
    "JobData",
    "DatasetData"
]

from dataclasses import dataclass

from zcommons.dataclass import DataMixin

from .config import *
from .meta import *


@dataclass()
class JobData(DataMixin):

    meta: JobMeta
    job_config: JobConfig
    train_config: TrainConfig
    aggregate_config: AggregateConfig

    @property
    def config(self):
        return {
            "job": self.job_config.to_json_dict(),
            "train": self.train_config.to_json_dict(),
            "aggregate_config": self.aggregate_config.to_json_dict()
        }

    def to_json_dict(self) -> dict:
        return {
            "meta": self.meta.to_json_dict(),
            "job_config": self.job_config.to_json_dict(),
            "train_config": self.train_config.to_json_dict(),
            "aggregate_config": self.aggregate_config
        }

    @classmethod
    def from_json_dict(cls, json_dict: dict) -> "JobData":
        return JobData(
            JobMeta.from_json_dict(json_dict["meta"]),
            JobConfig.from_json_dict(json_dict["job_config"]),
            TrainConfig.from_json_dict(json_dict["train_config"]),
            AggregateConfig.from_json_dict(json_dict["aggregate_config"])
        )


@dataclass()
class DatasetData(DataMixin):

    meta: DatasetMeta
    dataset_config: DatasetConfig

    @property
    def config(self):
        return {
            "dataset": self.dataset_config.to_json_dict()
        }

    def to_json_dict(self) -> dict:
        return {
            "meta": self.meta.to_json_dict(),
            "dataset_config": self.dataset_config.to_json_dict()
        }

    @classmethod
    def from_json_dict(cls, json_dict: dict) -> "DataMixin":
        return DatasetData(
            DatasetMeta.from_json_dict(json_dict["meta"]),
            DatasetConfig.from_json_dict(json_dict["dataset_config"])
        )
