#   Copyright 2020 The GFL Authors. All Rights Reserved.
#   #
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#   #
#       http://www.apache.org/licenses/LICENSE-2.0
#   #
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import os
import unittest

from gfl.core.db import init_sqlite
from gfl.core.fs import FS
from gfl.core.node import GflNode
from gfl.runtime.config import GflConfig
from gfl.runtime.manager.server_manager import ServerManager


class ServerManagerTest(unittest.TestCase):

    """
            if isinstance(config, GflConfig):
            self.__config = config
        elif isinstance(config, str):
            self.__config = GflConfig.load(config)
        else:
            self.__config = GflConfig(config)
        self.__fs.init(overwrite)
        self.node.save(self.__fs.path.key_file())
        self.config.save(self.__fs.path.config_file())
        init_sqlite(self.__fs.path.sqlite_file())
    """
    def setUp(self) -> None:
        home = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server-manager-home")
        fs = FS(home)
        node = GflNode.global_instance()
        config = GflConfig()
        # init
        fs.init(overwrite=True)
        node.save(fs.path.key_file())
        config.save(fs.path.config_file())
        init_sqlite(fs.path.sqlite_file())

        self.server_manager = ServerManager(fs, node, config)