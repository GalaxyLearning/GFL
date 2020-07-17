# Copyright (c) 2020 GalaxyLearning Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil

import ipfshttpclient

from gfl.settings import ipfs_url, TEMP_DIR_PATH
from gfl.utils.exception import IpfsException
from gfl.utils.json_utils import JsonUtil


class IpfsUtils(object):

    ipfs_client = ipfshttpclient.connect(ipfs_url, session=True)

    @classmethod
    def upload_file(cls, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError()
        return cls.ipfs_client.add(file_path)["Hash"]

    @classmethod
    def download_file(cls, ipfs_hash):
        try:
            cls.ipfs_client.get(ipfs_hash)
            temp_path = os.path.join(TEMP_DIR_PATH, ipfs_hash)
            if os.path.exists(temp_path):
                os.remove(temp_path)
            shutil.move(ipfs_hash, temp_path)
            return temp_path
        except:
            raise IpfsException()

    @classmethod
    def upload_json(cls, obj):
        return cls.upload_str(JsonUtil.to_json(obj))

    @classmethod
    def download_json(cls, ipfs_hash, obj_type):
        return JsonUtil.from_json(cls.download_str(ipfs_hash), obj_type)

    @classmethod
    def upload_str(cls, s):
        return cls.ipfs_client.add_str(s)

    @classmethod
    def download_str(cls, ipfs_hash):
        return cls.ipfs_client.cat(ipfs_hash)

