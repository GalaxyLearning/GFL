import os
import shutil

from ..shell import startup as shell_startup
from .app import GflApplication
from .config import GflConfig
from .constants import GFL_CONFIG_FILENAME
from .log import init_logging
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
    if force:
        __clean(home)
    os.makedirs(home, exist_ok=True)
    config = GflConfig(gfl_config)
    init_logging(config.log.level, os.path.join(home, config.log.path))

    app = GflApplication(home, config)
    app.init()


def gfl_start(home,
              no_webui,
              no_daemon,
              shell):
    if not home:
        home = default_home_path()
    config_path = os.path.join(home, GFL_CONFIG_FILENAME)
    if not os.path.exists(config_path):
        raise ValueError(f"{home} is not a valid gfl home path.")
    config = GflConfig(config_path)
    if no_webui is not None:
        config.app.http_webui_enabled = no_webui
    if shell is not None:
        config.app.shell_type = shell
    init_logging(config.log.level, os.path.join(home, config.log.path))

    if no_daemon is None:
        no_daemon = False
    app = GflApplication(home, config)
    app.start(not no_daemon)


def gfl_attach(shell_type,
               home,
               http_ip,
               http_port):
    shell_startup(shell_type, home, http_ip, http_port)
