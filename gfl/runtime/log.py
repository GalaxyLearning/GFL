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

import logging.config
import os
from logging import Filter

import zcommons as zc


class ColorFilter(Filter):

    colors = {
        logging.DEBUG: zc.FORE_CYAN,
        logging.INFO: zc.FORE_GREEN,
        logging.WARN: zc.FORE_YELLOW,
        logging.WARNING: zc.FORE_YELLOW,
        logging.ERROR: zc.FORE_RED,
        logging.CRITICAL: zc.FORE_MAGENTA
    }

    def __init__(self):
        super(ColorFilter, self).__init__()

    def filter(self, record) -> bool:
        color = self.colors.get(record.levelno, None)
        if color:
            record.levelname = f"{color}{record.levelname}{' ' * (8 - len(record.levelname))}{zc.FORE_RESET}"
        return True


def logging_config(log_level="INFO", log_root="logs", color=True, terminal_only=False):
    if isinstance(log_level, int):
        log_level = {
            logging.DEBUG: "DEBUG",
            logging.INFO: "INFO",
            logging.WARNING: "WARNING",
            logging.WARN: "WARNING",
            logging.ERROR: "ERROR",
            logging.CRITICAL: "CRITICAL"
        }[log_level]
    log_root = os.path.abspath(log_root)

    handlers = {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "common",
            "stream": "ext://sys.stdout",
            "filters": ["color"] if color else []
        }
    }
    if not terminal_only:
        handlers.update({
            "file": {
                "class": "logging.FileHandler",
                "level": "INFO",
                "formatter": "common",
                "filename": os.path.join(log_root, "root.log")
            },
            "core": {
                "class": "logging.FileHandler",
                "level": "INFO",
                "formatter": "common",
                "filename": os.path.join(log_root, "core.log")
            },
            "error": {
                "class": "logging.FileHandler",
                "level": "ERROR",
                "formatter": "common",
                "filename": os.path.join(log_root, "error.log")
            }
        })

    loggers = {
        "gfl": {
            "level": log_level,
            "handlers": ["console", "file"] if not terminal_only else ["console"],
            "propagate": "no"
        }
    }
    if not terminal_only:
        loggers.update({
            "fedflow.core": {
                "level": log_level,
                "handlers": ["core"],
                "propagate": "yes"
            }
        })

    return {
        "version": 1,
        "formatters": {
            "common": {
                "format": "%(asctime)s %(process)5d %(name)16s [%(levelname)-5s] %(message)s"
            }
        },
        "filters": {
            "color": {
                "()": ColorFilter
            }
        },
        "handlers": handlers,
        "loggers": loggers
    }


def update_logging_config(log_level="INFO", log_root="logs", color=True, terminal_only=False):
    config = logging_config(log_level, log_root, color, terminal_only)
    logging.config.dictConfig(config)


def set_level(log_level):
    update_logging_config(log_level=log_level)


def set_root(log_root):
    update_logging_config(log_root=log_root)


def set_color(use_color):
    update_logging_config(color=use_color)


update_logging_config(terminal_only=True)
