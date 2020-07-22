import os
import json
from gfl.settings import DATA_DIR_PATH


class LocalStorage(object):

    runtime_config_path = os.path.join(DATA_DIR_PATH, "runtime.conf")
    cache = {}

    if os.path.exists(runtime_config_path):
        with open(runtime_config_path, "r") as f:
            cache = json.loads(f.read())

    @classmethod
    def get_property(cls, key: str):
        return cls.cache.get(key)

    @classmethod
    def set_property(cls, key: str, value: str):
        cls.cache[key] = value
        with open(cls.runtime_config_path, "w") as f:
            f.write(json.dumps(cls.cache))

    @classmethod
    def remove_property(cls, key: str):
        del cls.cache[key]
        with open(cls.runtime_config_path, "w") as f:
            f.write(json.dumps(cls.cache))
