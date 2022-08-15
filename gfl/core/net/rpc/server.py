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

from concurrent import futures

import grpc

from gfl.data import *
import google.protobuf.wrappers_pb2 as wrappers_pb2
import gfl.core.data_pb2 as data_pb2
import gfl.core.net.rpc.gfl_pb2_grpc as gfl_pb2_grpc
from gfl.core.node import GflNode
from gfl.runtime.manager.server_manager import ServerManager


class GflServicer(gfl_pb2_grpc.GflServicer):

    def __init__(self, manager: ServerManager):
        super(GflServicer, self).__init__()
        self._manager = manager
        self.nodes = {}

    def SendNodeInfo(self, request, context):
        self.nodes[context.peer()] = GflNode(address=request.address, pub_key=request.pub_key)
        return wrappers_pb2.BoolValue(value=True)

    def GetPubKey(self, request, context):
        pub_key = ""
        for _, node in self.nodes.items():
            if node.address == request.address.value:
                pub_key = node.pub_key
                break
        return wrappers_pb2.StringValue(value=pub_key)

    def SendHealth(self, request, context):
        cp = ComputingResource()
        cp.mem_used = request.computing_power.mem_used
        cp.mem_total
        node = self.nodes[context.peer()]
        self._manager.update_resource(node.address, request.running_job_count, cp)
        return wrappers_pb2.BoolValue(value=True)

    def GetNetComputingPower(self, request, context):
        return self._manager.get_net_resource()

    def GetJobComputingPower(self, request, context):
        return self._manager.get_node_resource(request.value)

    def FetchJobMetas(self, request, context):
        pass

    def FetchJob(self, request, context):
        pass

    def PushJob(self, request, context):
        pass

    def JoinJob(self, request, context):
        pass

    def FetchDatasetMetas(self, request, context):
        pass

    def FetchDataset(self, request, context):
        pass

    def PushDataset(self, request, context):
        pass

    def FetchParams(self, request, context):
        pass

    def PushParams(self, request, context):
        pass


def startup(manager: ServerManager):
    print(f"Startup gRPC")
    rpc_config = manager.config.node.rpc
    bind_host, bind_port = rpc_config.server_host, rpc_config.server_port
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=rpc_config.max_workers))
    gfl_pb2_grpc.add_GflServicer_to_server(GflServicer(manager), server)
    server.add_insecure_port(f"{bind_host}:{bind_port}")
    server.start()
    res = server.wait_for_termination()
    print(f"Res: {res}")
