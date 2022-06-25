import os

from .constants import ENV_GFL_HOME, GFL_CONFIG_FILENAME


def default_home_path():
    home = os.getenv(ENV_GFL_HOME, None)
    if home is None:
        home = os.path.join(os.path.expanduser("~"), ".gfl_p")
    return home


def default_config_path():
    config_path = os.path.join(default_home_path(), GFL_CONFIG_FILENAME)
    if os.path.exists(config_path):
        return config_path
    else:
        return None


def check_home(home, is_init, is_attach):
    if home is None:
        if is_attach:
            return
        home = default_home_path()
    if not is_init:
        config_path = os.path.join(home, GFL_CONFIG_FILENAME)
        if not os.path.exists(config_path):
            raise ValueError(f"Cannot find gfl_config.yaml file in gfl_p home path.")
