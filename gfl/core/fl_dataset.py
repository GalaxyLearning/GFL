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
    "FLDataset"
]

import json
import os
import shutil

import zcommons as zc

from gfl.core.fs.path import DatasetPath
from gfl.data import *
from gfl.utils import ZipUtils


class FLDataset(object):

    def __init__(self, dataset_path: DatasetPath, dataset_data: DatasetData):
        super(FLDataset, self).__init__()
        self._path = dataset_path
        self._data = dataset_data

    @property
    def data(self):
        return self._data

    @property
    def id(self):
        return self._data.meta.id

    @classmethod
    def load(cls, dataset_path: DatasetPath, dataset_id: str):
        path = dataset_path.dataset_dir(dataset_id)
        if not os.path.exists(path) or not os.path.isdir(path) or len(os.listdir(path)) == 0:
            raise ValueError(f"Dataset({dataset_id}) not exists.")
        with open(dataset_path.meta_file(dataset_id), "r") as f:
            dataset_meta = zc.dataclass.asobj(DatasetMeta, json.loads(f.read()))
        with open(dataset_path.config_file(dataset_id), "r") as f:
            config = json.loads(f.read())
            dataset_config = config["dataset"]
        dataset_data = DatasetData(
            meta=dataset_meta,
            dataset_config=dataset_config
        )
        return FLDataset(dataset_path, dataset_data)

    def save(self, package_data: bytes, overwrite=False):
        path = self._path.dataset_dir(self.id)
        if os.path.exists(path) and os.path.isdir(path) and len(os.listdir(path)) > 0:
            if overwrite:
                shutil.rmtree(path)
            else:
                raise ValueError(f"Dataset({self.id}) exists.")
        os.makedirs(path, exist_ok=True)
        with open(self._path.meta_file(self.id), "w") as f:
            f.write(json.dumps(zc.dataclass.asdict(self._data.meta)))
        with open(self._path.config_file(self.id), "w") as f:
            f.write(json.dumps(zc.dataclass.asdict(self._data.config)))
        ZipUtils.extract_data(package_data, self._path.module_dir(self.id))
