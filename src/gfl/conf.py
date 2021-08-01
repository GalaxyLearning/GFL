import logging.config
import os
import shutil
import tempfile
import warnings

import yaml

from gfl.utils import PathUtils


os_tempdir = tempfile.gettempdir()
gfl_tempdir = PathUtils.join(os_tempdir, "gfl")
if os.path.exists(gfl_tempdir):
    os.makedirs(gfl_tempdir, exist_ok=True)


class GflConfMetadata(type):

    @property
    def home_dir(cls):
        return cls._GflConf__home_dir

    @home_dir.setter
    def home_dir(cls, value):
        cls._GflConf__home_dir = PathUtils.abspath(value)
        cls._GflConf__data_dir = PathUtils.join(value, "data")
        cls._GflConf__logs_dir = PathUtils.join(value, "logs")
        cls._GflConf__cache_dir = PathUtils.join(value, "cache")

    @property
    def data_dir(cls):
        return cls._GflConf__data_dir

    @property
    def logs_dir(cls):
        return cls._GflConf__logs_dir

    @property
    def cache_dir(cls):
        return cls._GflConf__cache_dir

    @property
    def temp_dir(cls):
        return gfl_tempdir


class GflConf(object, metaclass=GflConfMetadata):

    # Parameters that can be modified at run time
    __props = {}
    # Parameters that are read from a configuration file and cannot be changed at run time
    __readonly_props = {}

    __home_dir = PathUtils.join(PathUtils.user_home_dir(), ".gfl")
    __data_dir = PathUtils.join(__home_dir, "data")
    __logs_dir = PathUtils.join(__home_dir, "logs")
    __cache_dir = PathUtils.join(__home_dir, "cache")
    __temp_dir = gfl_tempdir

    @classmethod
    def load(cls) -> None:
        """
        load config properties from disk file.

        :return:
        """
        base_config_path = PathUtils.join(PathUtils.src_root_dir(), "resources", "config.yaml")
        with open(base_config_path) as f:
            cls.__readonly_props = yaml.load(f, Loader=yaml.SafeLoader)

        path = PathUtils.join(cls.home_dir, "config.yaml")
        if os.path.exists(path):
            with open(path) as f:
                config_data = yaml.load(f, Loader=yaml.SafeLoader)
                cls.__readonly_props.update(config_data)

        if os.path.exists(cls.logs_dir):
            cls.load_logging_config()
        else:
            warnings.warn("cannot found logs dir.")

    @classmethod
    def load_logging_config(cls) -> None:
        """

        """
        logging_config_path = PathUtils.join(PathUtils.src_root_dir(), "resources", "logging.yaml")
        with open(logging_config_path) as f:
            text = f.read().replace("{logs_root}", GflConf.logs_dir)
            data = yaml.load(text, yaml.SafeLoader)

        if cls.get_property("debug"):
            data["root"]["level"] = "DEBUG"
            data["loggers"]["gfl"]["level"] = "DEBUG"

        logging.config.dictConfig(data)

    @classmethod
    def generate_config(cls, path: str = None) -> None:
        """
        generate config file in ``path``.

        :param path: the config file path, if it's None, will be replaced by './config.yaml'.
        :return:
        """
        if path is None:
            path = "config.yaml"
        src_path = PathUtils.join(PathUtils.src_root_dir(), "resources", "config.yaml")
        shutil.copy(src_path, path)

    @classmethod
    def set_config(cls, d: dict) -> None:
        """
        Batch update config properties. Generally, this method is not recommend.

        :param d: a dict represent config properties.
        :return:
        """
        cls.__props.update(d.copy())

    @classmethod
    def get_property(cls, key, default=None):
        """
        Get the value of readonly parameters.

        :param key: a string of the key to get the value
        :param default: return value if key not found
        """
        op_res, val = cls.__get_from_dict(cls.__props,
                                          cls.__split_key(key),
                                          default)
        if op_res:
            return val
        return cls.__get_from_dict(cls.__readonly_props,
                                   cls.__split_key(key),
                                   default)[1]

    @classmethod
    def set_property(cls, key, value):
        """
        Set parameters at run time.

        :param key:
        :param value:
        :return:
        """
        cls.__set_to_dict(cls.__props, cls.__split_key(key), value)

    @classmethod
    def remove_property(cls, key):
        cls.__remove_from_dict(cls.__props, cls.__split_key(key))

    @classmethod
    def __split_key(cls, key: str):
        if key is None or key.strip() == "":
            raise ValueError("key cannot be none or empty.")
        return key.split(".")

    @classmethod
    def __exists_in_dict(cls, d: dict, k_seq: list):
        if k_seq is None or len(k_seq) == 0:
            return False
        for k in k_seq:
            if k in d:
                d = d[k]
            else:
                return False
        return True

    @classmethod
    def __get_from_dict(cls, d: dict, k_seq: list, default=None):
        if k_seq is None or len(k_seq) == 0:
            raise ValueError("key cannot be none or empty")
        for k in k_seq:
            if k in d:
                d = d[k]
            else:
                return False, default
        return True, d

    @classmethod
    def __remove_from_dict(cls, d: dict, k_seq: list):
        if k_seq is None or len(k_seq) == 0:
            raise ValueError("key cannot be none or empty")
        for k in k_seq[:-1]:
            if k not in d:
                return False
            d = d[k]
        try:
            del d[k_seq[-1]]
            return True
        except:
            return False


    @classmethod
    def __set_to_dict(cls, d: dict, k_seq: list, value):
        if k_seq is None or len(k_seq) == 0:
            raise ValueError("key cannot be none or empty")
        for k in k_seq[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]
        d[k_seq[-1]] = value
