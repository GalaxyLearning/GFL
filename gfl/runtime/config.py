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

import json
import yaml
from traits.api import HasTraits, Str, Int, Bool, Instance


def _load_dict(path, ext):
    if ext == "yaml":
        return yaml.load(path, Loader=yaml.SafeLoader)
    elif ext == "json":
        with open(path) as f:
            return json.loads(f.read(), encoding="utf8")
    else:
        raise ValueError(f"Expected yaml or json ext. Received: {ext}")


def _check_int_between(_v, _min, _max, msg):
    if not isinstance(_v, int) or _v < _min or _v > _max:
        raise ValueError(f"")


class LogConfig(HasTraits):
    level = Str
    root = Str

    def __init__(self, config_dict:dict=None):
        super(LogConfig, self).__init__()
        if config_dict is None:
            config_dict = {}
        self.level = config_dict.get("level", "INFO")
        self.path = config_dict.get("root", "logs")

    @property
    def config_dict(self):
        return {
            "level": self.level,
            "root": self.root
        }


class HttpRpcConfig(HasTraits):
    enabled = Bool
    as_server = Bool
    server_host = Str
    server_port = Int
    max_workers = Int

    def __init__(self, config_dict:dict=None):
        super(HttpRpcConfig, self).__init__()
        if config_dict is None:
            config_dict = {}
        self.enabled = config_dict.get("enabled", False)
        self.as_server = config_dict.get("as_server", False)
        self.server_host = config_dict.get("server_host", "127.0.0.1")
        self.server_port = config_dict.get("server_port", 10702)
        self.max_workers = config_dict.get("max_workers", 3)

    @property
    def config_dict(self):
        return {
            "enabled": self.enabled,
            "as_server": self.as_server,
            "server_host": self.server_host,
            "server_port": self.server_port
        }


class EthConfig(HasTraits):
    enabled = Bool
    eth_host = Str
    eth_port = Int
    contract_address = Str

    def __init__(self, config_dict:dict=None):
        super(EthConfig, self).__init__()
        if config_dict is None:
            config_dict = {}
        self.enabled = config_dict.get("enabled", False)
        self.eth_host = config_dict.get("eth_host", "127.0.0.1")
        self.eth_port = config_dict.get("eth_port", 8545)
        self.contract_address = config_dict.get("contract_address", "")

    @property
    def config_dict(self):
        return {
            "enabled": self.enabled,
            "eth_host": self.eth_host,
            "eth_port": self.eth_port,
            "contract_address": self.contract_address
        }


class NodeConfig(HasTraits):
    http = Instance(HttpRpcConfig)
    rpc = Instance(HttpRpcConfig)
    eth = Instance(EthConfig)

    def __init__(self, config_dict: dict = None):
        super(NodeConfig, self).__init__()
        if config_dict is None:
            config_dict = {}
        self.http = HttpRpcConfig(config_dict.get("http", {}))
        self.rpc = HttpRpcConfig(config_dict.get("rpc", {}))
        self.eth = EthConfig(config_dict.get("eth", {}))

    @property
    def config_dict(self):
        return {
            "http": self.http.config_dict,
            "rpc": self.rpc.config_dict,
            "eth": self.eth.config_dict
        }


class AppConfig(HasTraits):
    shell = Str

    def __init__(self, config_dict):
        super(AppConfig, self).__init__()
        if config_dict is None:
            config_dict = {}
        self.shell = config_dict.get("shell", "ipython")

    @property
    def config_dict(self):
        return {
            "shell": self.shell
        }


class GflConfig(HasTraits):

    app = Instance(AppConfig)
    node = Instance(NodeConfig)
    log = Instance(LogConfig)

    def __init__(self, config_dict: dict = None):
        super(GflConfig, self).__init__()
        if config_dict is None:
            config_dict = {}
        self.app = AppConfig(config_dict.get("app", {}))
        self.node = NodeConfig(config_dict.get("node", {}))
        self.log = LogConfig(config_dict.get("log", {}))

    @property
    def config_dict(self):
        return {
            "app": self.app.config_dict,
            "node": self.node.config_dict,
            "log": self.log.config_dict
        }

    def save(self, path):
        with open(path, "w") as f:
            f.write(json.dumps(self.config_dict, indent=4))

    @classmethod
    def load(cls, path, ext="json"):
        config_dict = _load_dict(path, ext)
        return GflConfig(config_dict)
