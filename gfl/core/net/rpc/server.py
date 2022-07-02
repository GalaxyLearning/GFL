from concurrent import futures

import grpc

import gfl.core.net.rpc.gfl_pb2_grpc as gfl_pb2_grpc
from gfl.runtime.manager import NodeManager


class GflServicer(gfl_pb2_grpc.GflServicer):

    def __init__(self, manager: NodeManager):
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
