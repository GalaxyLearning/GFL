"""
key.json

data
    jobs
        <job_id>
            meta.json
            config.json
            job.sqlite
            fl_job
                __init__.py
            params
                <step>
                    train
                        <address>
                    aggregate
                        <address>
            metrics
            reports
    datasets
        <dataset_id>
            meta.json
            config.json
            fl_dataset
                __init__.py
logs
"""
__all__ = [
    "PathLike",
    "Path",
    "JobPath",
    "DatasetPath"
]

import os
import shutil
from functools import wraps
from pathlib import PurePosixPath


class PathLike(os.PathLike):
    # TODO: Add protocol support, such as file://, ipfs://, grpc://

    def __init__(self, path):
        super(PathLike, self).__init__()
        self.path = path

    def abspath(self):
        return os.path.abspath(self.path)

    def exists(self):
        return os.path.exists(self.path)

    def makedirs(self, exist_ok=True):
        os.makedirs(self.path, exist_ok=exist_ok)

    def rm(self):
        shutil.rmtree(str(self))

    def as_posix(self):
        posix_path = PurePosixPath(self.path)
        return posix_path.as_posix()

    def __fspath__(self):
        return str(self)

    def __repr__(self):
        return self.path

    def __str__(self):
        return self.path


def path_like(func):

    @wraps(func)
    def wrapper(*args, **kwargs) -> PathLike:
        return PathLike(func(*args, **kwargs))

    return wrapper


class JobPath(object):

    def __init__(self, job_root):
        super(JobPath, self).__init__()
        self.__job_root = job_root

    @path_like
    def root_dir(self):
        return self.__job_root

    @path_like
    def meta_file(self, job_id):
        return os.path.join(self.__job_root, job_id, "meta.json")

    @path_like
    def sqlite_file(self, job_id):
        return os.path.join(self.__job_root, job_id, "job.sqlite")

    @path_like
    def config_file(self, job_id):
        return os.path.join(self.__job_root, job_id, "config.sqlite")

    @path_like
    def module_dir(self, job_id):
        return os.path.join(self.__job_root, job_id, "fl_job")

    @path_like
    def params_dir(self, job_id):
        return os.path.join(self.__job_root, job_id, "params")

    @path_like
    def metrics_dir(self, job_id):
        return os.path.join(self.__job_root, job_id, "metrics")

    @path_like
    def reports_dir(self, job_id):
        return os.path.join(self.__job_root, job_id, "reports")

    @path_like
    def train_params_dir(self, job_id, step, address):
        return os.path.join(self.__job_root, job_id, "params", str(step), "train", address)

    @path_like
    def aggregate_params_dir(self, job_id, step, address):
        return os.path.join(self.__job_root, job_id, "params", str(step), "aggregate", address)


class DatasetPath(object):

    def __init__(self, dataset_root):
        super(DatasetPath, self).__init__()
        self.__dataset_root = dataset_root

    @path_like
    def root_dir(self):
        return self.__dataset_root

    @path_like
    def meta_file(self, dataset_id):
        return os.path.join(self.__dataset_root, dataset_id, "meta.json")

    @path_like
    def config_file(self, dataset_id):
        return os.path.join(self.__dataset_root, dataset_id, "config.json")

    @path_like
    def module_dir(self, dataset_id):
        return os.path.join(self.__dataset_root, dataset_id, "fl_dataset")


class Path(object):

    def __init__(self, home):
        super(Path, self).__init__()
        self.__home = os.path.abspath(home)
        self.__config_file = os.path.join(self.__home, "config.json")
        self.__key_file = os.path.join(self.__home, "key.json")
        self.__data_dir = os.path.join(self.__home, "data")
        self.__logs_dir = os.path.join(self.__home, "logs")
        self.__job_path = JobPath(os.path.join(self.__data_dir, "jobs"))
        self.__dataset_path = DatasetPath(os.path.join(self.__data_dir, "datasets"))

    @path_like
    def home(self):
        return self.__home

    @path_like
    def config_file(self):
        return self.__config_file

    @path_like
    def key_file(self):
        return self.__key_file

    @path_like
    def data_dir(self):
        return self.__data_dir

    @path_like
    def logs_dir(self):
        return self.__logs_dir

    @property
    def job(self):
        return self.__job_path

    @property
    def dataset(self):
        return self.__dataset_path
