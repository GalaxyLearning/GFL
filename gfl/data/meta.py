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
    "JobMeta",
    "DatasetMeta"
]

from dataclasses import dataclass
from typing import List

from zcommons.dataclass import DataMixin

from .constants import *


@dataclass()
class Metadata(DataMixin):

    id: str = None
    owner: str = None
    create_time: int = None
    content: str = None


@dataclass()
class JobMeta(Metadata, DataMixin):

    status: JobStatus = JobStatus.NEW
    datasets: List[str] = None

    def to_json_dict(self) -> dict:
        return {
            "id": self.id,
            "owner": self.owner,
            "create_time": self.create_time,
            "content": self.content,
            "status": self.status.value,
            "datasets": self.datasets
        }

    @classmethod
    def from_json_dict(cls, json_dict: dict) -> "JobMeta":
        return JobMeta(
            json_dict["id"],
            json_dict["owner"],
            json_dict["create_time"],
            json_dict["content"],
            JobStatus(json_dict["status"]),
            json_dict["datasets"]
        )


@dataclass()
class DatasetMeta(Metadata, DataMixin):

    type: DatasetType = DatasetType.IMAGE
    status: DatasetStatus = DatasetStatus.NEW
    size: int = 0
    used_cnt: int = 0
    request_cnt: int = 0

    def to_json_dict(self) -> dict:
        return {
            "id": self.id,
            "owner": self.owner,
            "create_time": self.create_time,
            "content": self.content,
            "type": self.type.value,
            "status": self.status.value,
            "size": self.size,
            "used_cnt": self.used_cnt,
            "request_cnt": self.request_cnt
        }

    @classmethod
    def from_json_dict(cls, json_dict: dict) -> "DatasetMeta":
        return DatasetMeta(
            json_dict["id"],
            json_dict["owner"],
            json_dict["create_time"],
            json_dict["content"],
            DatasetType(json_dict["type"]),
            DatasetStatus(json_dict["status"]),
            json_dict["size"],
            json_dict["used_cnt"],
            json_dict["request_cnt"]
        )
