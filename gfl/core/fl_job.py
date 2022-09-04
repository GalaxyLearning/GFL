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
    "FLJob"
]

import json
import os
import shutil

import zcommons as zc

from gfl.core.fs.path import JobPath
from gfl.data import *
from gfl.utils import ZipUtils


class FLJob(object):

    def __init__(self, job_path: JobPath, job_data: JobData = None):
        super(FLJob, self).__init__()
        self._path = job_path
        self._data = job_data

    @property
    def data(self):
        return self._data

    @property
    def id(self):
        return self._data.meta.id

    @classmethod
    def load(cls, job_path: JobPath, job_id: str):
        path = job_path.job_dir(job_id)
        if not os.path.exists(path) or not os.path.isdir(path) or len(os.listdir(path)) == 0:
            raise ValueError(f"Job({job_id}) not exists")
        with open(job_path.meta_file(job_id), "r") as f:
            job_meta = zc.dataclass.asobj(JobMeta, json.loads(f.read()))
        with open(job_path.config_file(job_id), "r") as f:
            config = json.loads(f.read())
            job_config = zc.dataclass.asobj(JobConfig, config["job"])
            train_config = zc.dataclass.asobj(TrainConfig, config["train"])
            aggregate_config = zc.dataclass.asobj(AggregateConfig, config["aggregate"])
        job_data = JobData(
            meta=job_meta,
            job_config=job_config,
            train_config=train_config,
            aggregate_config=aggregate_config
        )
        return FLJob(job_path, job_data)

    def save(self, package_data: bytes, overwrite=False):
        path = self._path.job_dir(self.id)
        if os.path.exists(path) and os.path.isdir(path) and len(os.listdir(path)) > 0:
            if overwrite:
                shutil.rmtree(path)
            else:
                raise ValueError(f"Job({self.id}) exists.")
        # make dirs
        os.makedirs(path, exist_ok=True)
        with open(self._path.meta_file(self.id), "w") as f:
            f.write(self._data.meta.to_json())
        with open(self._path.config_file(self.id), "w") as f:
            f.write(json.dumps(self._data.config))
        os.makedirs(self._path.params_dir(self.id))
        os.makedirs(self._path.metrics_dir(self.id))
        os.makedirs(self._path.reports_dir(self.id))
        ZipUtils.extract_data(package_data, self._path.module_dir(self.id))
