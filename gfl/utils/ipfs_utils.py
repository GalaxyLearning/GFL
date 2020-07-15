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
import ipfshttpclient
from pfl.exceptions.fl_expection import PFLException



class IpfsUtils(object):

    def __init__(self):
        pass

    @staticmethod
    def init_ipfs_api_instance(url=None):
        if url is None:
            raise PFLException("need url parameter")

        api = ipfshttpclient.connect(url, session=True)

        return api

    @staticmethod
    def uploadFile(file_path=None, api=None):
        if file_path is None or api is None:
            raise PFLException("parameter error")
        res = api.add(file_path)
        return res

    @staticmethod
    def downloadFile(file_ipfs_hash=None, file_name=None, api=None):

        if file_ipfs_hash is None or api is None or file_name is None:
            raise PFLException("parameter error")
        api.get(file_ipfs_hash)
        if os.path.exists(file_ipfs_hash):
            os.rename(file_ipfs_hash, file_name)
            return True
        return False


if __name__ == "__main__":
    ipfs_api = IpfsUtils.init_ipfs_api_instance(url='/ip4/10.5.18.241/tcp/5001/http')

    # res Hash: QmNprJ78ovcUuGMoMFiihK7GBpCmH578JU8hm43uxYQtBw
    # res = IpfsUtils.uploadFile("/Users/huyifan/Documents/PFL/LICENSE", ipfs_api)
    # print(res)
    print(os.path.abspath("."))
    res2 = IpfsUtils.downloadFile("QmNprJ78ovcUuGMoMFiihK7GBpCmH578JU8hm43uxYQtBw", "LICENSE2", ipfs_api)
    print(res2)