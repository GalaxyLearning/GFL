
from .ipc_adapter import IpcAdapter


class Node(object):

    def __init__(self, comm_provider: IpcAdapter):
        super(Node, self).__init__()
