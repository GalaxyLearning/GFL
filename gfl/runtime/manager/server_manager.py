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

import os
import typing

import zcommons as zc

from ..config import GflConfig
from gfl.core.db import DB
from gfl.core.fs import FS
from gfl.core.fl_dataset import FLDataset
from gfl.core.fl_job import FLJob
from gfl.core.node import GflNode
from gfl.data import *
from gfl.runtime.manager.resource_manager import ResourceManager


class ServerManager(object):

    def __init__(self, fs: FS, node: GflNode, config: GflConfig):
        super(ServerManager, self).__init__()
        self.__fs = fs
        self.__node = node
        self.__config = config
        self.__db = DB(fs.path.sqlite_file())
        self.__resource_manager = ResourceManager()

    @property
    def config(self) -> GflConfig:
        return self.__config

    def update_resource(self, node_address, computing_resource):
        self.__resource_manager.update_resource(node_address, computing_resource)

    def get_node_resource(self, node_address):
        return self.__resource_manager.get_resource(node_address)

    def get_net_resource(self):
        return self.__resource_manager.get_net_resource()

    def push_job(self, job_data: JobData, package_data: bytes):
        job = FLJob(job_path=self.__fs.path.job,
                    job_data=job_data)
        job.save(package_data, overwrite=False)
        # self.__db.add_job(job_data)

    def push_dataset(self, dataset_data: DatasetData, package_data: bytes):
        dataset = FLDataset(dataset_path=self.__fs.path.dataset,
                            dataset_data=dataset_data)
        dataset.save(package_data, overwrite=False)
        self.__db.add_dataset(dataset_data)

    def fetch_job_metas(self, status) -> typing.List[JobMeta]:
        pass

    def fetch_job(self, job_id: str) -> JobData:
        job = FLJob.load(self.__fs.path.job, job_id)
        return job.data

    def fetch_dataset_metas(self, status) -> typing.List[DatasetMeta]:
        pass

    def fetch_dataset(self, dataset_id) -> DatasetData:
        dataset = FLDataset.load(self.__fs.path.dataset, dataset_id)
        return dataset.data

    def fetch_params(self):
        pass

    def push_params(self):
        pass
