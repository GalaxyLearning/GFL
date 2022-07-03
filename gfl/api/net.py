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

import google.protobuf.wrappers_pb2 as wrappers_pb2
import gfl.core.data_pb2 as data_pb2
import gfl.core.net.rpc.gfl_pb2 as gfl_pb2
from gfl.core.fs import FS
from gfl.core.net.rpc.client import build_client
from gfl.core.node import GflNode
from gfl.runtime.config import GflConfig


class Net(object):

    def __init__(self, home):
        super(Net, self).__init__()
        self.__fs = FS(home)
        self.__config = GflConfig.load(self.__fs.path.config_file())
        self.__node = GflNode.load_node(self.__fs.path.key_file())
        self.__client = None
        self.__init_client()

    def __init_client(self):
        self.__client = build_client(self.__config.node.rpc.server_host, self.__config.node.rpc.server_port)
        self.__client.SendNodeInfo(gfl_pb2.NodeInfo(
            address=self.__node.address,
            pub_key=self.__node.pub_key
        ))

    def get_pub_key(self, address):
        resp = self.__client.GetPubKey(wrappers_pb2.StringValue(value=address))
        print(f"Resp: {resp.value}")
