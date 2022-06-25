import logging.config
import os

import yaml

from .constants import LOGGING_YAML_PATH


def __replace_filepath(config: dict, root_path: str):
    for k, v in config.keys():
        v = config[k]
        if k == "filename" and isinstance(v, str):
            config[k] = v.replace("{logs_root}", root_path)
        elif isinstance(v, dict):
            __replace_filepath(v, root_path)


def init_logging(log_level, log_path):
    config = yaml.load(LOGGING_YAML_PATH, Loader=yaml.SafeLoader)
    # replace root path
    log_path = os.path.abspath(log_path)
    __replace_filepath(config, log_path)
    # replace log level
    for _, logger in config.get("loggers", {}).items():
        logger["level"] = log_level

    logging.config.dictConfig(config)
