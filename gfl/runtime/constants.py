import os


__FILE_ABSPATH = os.path.abspath(__file__)
GFL_PATH = os.path.dirname(os.path.dirname(__FILE_ABSPATH))
GFL_CONFIG_FILENAME = "gfl_config.yaml"
CONFIG_YAML_PATH = os.path.join(GFL_PATH, "resources", GFL_CONFIG_FILENAME)
LOGGING_YAML_PATH = os.path.join(GFL_PATH, "resources", "logging.yaml")

# os envs
ENV_GFL_HOME = "GFL_HOME"

# config keys
KEY_LOG_LEVEL = "log.level"
KEY_LOG_PATH = "log.path"
KEY_APP_SOCKET_BIND_IP = "app.socket.bind_ip"
KEY_APP_SOCKET_BIND_PORT = "app.socket.bind_port"
KEY_APP_HTTP_ENABLED = "app.http.enabled"
KEY_APP_HTTP_WEBUI_ENABLED = "app.http.webui_enabled"
KEY_APP_HTTP_BIND_IP = "app.http.bind_ip"
KEY_APP_HTTP_BIND_PORT = "app.http.bind_port"
KEY_APP_HTTP_ALLOW_CMD = "app.http.allow_cmd"
KEY_APP_SHELL_TYPE = "app.shell.type"
