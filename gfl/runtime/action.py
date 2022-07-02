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

import os
import shutil

from ..shell import startup as shell_startup
from .app import GflApplication
from .config import GflConfig
from .log import update_logging_config
from .utils import default_home_path


def __clean(path):
    if not os.path.exists(path):
        return
    if not os.path.isdir(path):
        raise ValueError(f"{path} is not a directory.")
    shutil.rmtree(path)


def gfl_init(home,
             gfl_config,
             force):
    if not home:
        home = default_home_path()
    if not gfl_config:
        raise ValueError(f"Expected a config file when init gfl.")
    config = GflConfig.load(gfl_config)
    update_logging_config(log_level=config.log.level, log_root=os.path.join(home, config.log.root))

    app = GflApplication(home)
    app.init(config, overwrite=force)


def gfl_start(home,
              no_webui,
              no_daemon,
              shell):
    if not home:
        home = default_home_path()

    app = GflApplication(home)
    app.start(not no_daemon)


def gfl_attach(shell_type,
               home,
               http_ip,
               http_port):
    shell_startup(shell_type, home, http_ip, http_port)
