import os

from gfl import GFL, Node
from gfl.ipc_adapter import IpcAdapter


def init_ipc_provider():
    node_ip = os.environ.get("__GFL_NODE_IP__", None)
    node_port = os.environ.get("__GFL_NODE_PORT__", None)

    provider = IpcAdapter(node_ip, node_port)
    return provider


def init_cmd():
    provider = init_ipc_provider()
    gfl = GFL(provider)
    node = Node(provider)
    return gfl, node


gfl, node = init_cmd()
