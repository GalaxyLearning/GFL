import grpc

import gfl.core.net.rpc.gfl_pb2_grpc as gfl_pb2_grpc


def build_client(host, port) -> gfl_pb2_grpc.GflStub:
    channel = grpc.insecure_channel(f"{host}:{port}")
    stub = gfl_pb2_grpc.GflStub(channel)
    return stub
