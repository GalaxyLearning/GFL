

from .config import GflConfig
from gfl.core.fs import FS
from gfl.core.node import GflNode


class GflApplication(object):

    def __init__(self, home):
        super(GflApplication, self).__init__()
        self.__fs = FS(home)
        self.__node = GflNode.global_instance()
        self.__config = GflConfig()

    @property
    def home(self) -> str:
        return self.__fs.home

    @property
    def config(self) -> GflConfig:
        return self.__config

    @property
    def fs(self) -> FS:
        return self.__fs

    @property
    def node(self) -> GflNode:
        return self.__node

    def init(self, config, *, overwrite):
        if isinstance(config, GflConfig):
            self.__config = config
        elif isinstance(config, str):
            self.__config = GflConfig.load(config)
        else:
            self.__config = GflConfig(config)
        self.__fs.init(overwrite)
        self.node.save(self.__fs.path.key_file())
        self.config.save(self.__fs.path.config_file())

    def start(self, daemon=False):
        self.__node = GflNode.load_node(self.__fs.path.key_file())
        self.__config = GflConfig.load(self.__fs.path.config_file())
        if self.__config.node.rpc.as_server:
            self.__start_server(daemon)
        else:
            self.__start_client(daemon)

    def __start_server(self, daemon):
        from ..core.net.rpc.server import startup
        from .manager.server_manager import ServerManager
        manager = ServerManager(self.__fs, self.__node, self.__config)
        startup(manager)

    def __start_client(self, daemon):
        from ..core.net.rpc.client import build_client
        from .manager.client_manager import ClientManager
        client = build_client(self.__config.node.rpc.server_host, self.__config.node.rpc.server_port)
        manager = ClientManager(self.__fs, self.__node, self.__config, client)
        manager.startup()
