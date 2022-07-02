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

import gfl.core.net.rpc.gfl_pb2_grpc as gfl_pb2_grpc
from gfl.runtime.manager.server_manager import ServerManager


class GflServicer(gfl_pb2_grpc.GflServicer):

    def __init__(self, manager: ServerManager):
        super(GflServicer, self).__init__()
        self._manager = manager

    def SendHealth(self, request, context):
        pass

    def GetNetComputingPower(self, request, context):
        pass

    def GetJobComputingPower(self, request, context):
        pass

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


def startup(manager):
    bind_host, bind_port, max_workers = "", 1, 1
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    gfl_pb2_grpc.add_GflServicer_to_server(GflServicer(manager), server)
    server.add_insecure_port(f"{bind_host}:{bind_port}")
    server.start()
