import os
from typing import Union

import hjson
import yaml
import zcommons as zc

from .constants import *


def _load_dict(path, ext):
    if ext == "yaml":
        return yaml.load(path, Loader=yaml.SafeLoader)
    elif ext == "json":
        return hjson.load(path, encoding="utf8")
    else:
        raise ValueError(f"Expected yaml or json ext. Received: {ext}")


def _check_int_between(_v, _min, _max, msg):
    if not isinstance(_v, int) or _v < _min or _v > _max:
        raise ValueError(f"")


class LogConfig(object):

    def __init__(self):
        super(LogConfig, self).__init__()
        self.level = None
        self.path = None


class AppConfig(object):

    def __init__(self):
        super(AppConfig, self).__init__()
        self.socket_bind_ip = None
        self.socket_bind_port = None

        self.http_enabled = None
        self.http_webui_enabled = None
        self.http_bind_ip = None
        self.http_bind_port = None
        self.http_allow_cmd = None

        self.shell_type = None


class GflConfig(object):

    def __init__(self, config: Union[str, dict] = None):
        super(GflConfig, self).__init__()
        if config is None:
            self._param_dict = _load_dict(CONFIG_YAML_PATH, "yaml")
        elif isinstance(config, dict):
            self._param_dict = config
        elif os.path.exists(config):
            self._param_dict = _load_dict(config, "yaml")
        else:
            raise ValueError(
                f"Expected a string path to an existing gfl_p config, or a dict. Received: {config}"
            )
        self.__config = zc.Config(self._param_dict)

        self.log = self.__init_log()
        self.app = self.__init_app()

    def __init_log(self):
        log_config = LogConfig()
        log_config.level = self.__config.get(KEY_LOG_LEVEL, "INFO")
        log_config.path = self.__config.get(KEY_LOG_PATH, "logs")
        return log_config

    def __init_app(self):
        app = AppConfig()
        app.socket_bind_ip = self.__config.get(KEY_APP_SOCKET_BIND_IP, "127.0.0.1")
        app.socket_bind_port = self.__config.get(KEY_APP_SOCKET_BIND_PORT, 10701)
        app.http_enabled = self.__config.get(KEY_APP_HTTP_ENABLED, True)
        app.http_webui_enabled = self.__config.get(KEY_APP_HTTP_WEBUI_ENABLED, False)
        app.http_bind_ip = self.__config.get(KEY_APP_HTTP_BIND_IP, "0.0.0.0")
        app.http_bind_port = self.__config.get(KEY_APP_HTTP_BIND_PORT, 10700)
        app.http_allow_cmd = self.__config.get(KEY_APP_HTTP_ALLOW_CMD, ["gfl_p", "node"])
        app.shell_type = self.__config.get(KEY_APP_SHELL_TYPE, "ipython")
        return app
