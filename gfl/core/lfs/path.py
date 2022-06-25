import os
import time
import threading

from gfl.conf import GflConf
from gfl.utils import PathUtils


class Path(object):

    _instances = {}
    _lock = threading.Lock()

    def __init__(self, id):
        super(Path, self).__init__()
        self.__id = id
        self.__last_access_time = time.time()

    def __new__(cls, id):
        if id is None:
            return object.__new__(cls)
        cls._lock.acquire()
        if len(cls._instances) > 127:
            cls.__clear()
        obj = cls._instances.get(id)
        if obj is None:
            obj = object.__new__(cls)
            cls._instances[id] = obj
        cls._lock.release()
        return obj

    @classmethod
    def __clear(cls):
        pop_id = ""
        earliest_time = time.time()
        for k, v in cls._instances.items():
            if v.__last_access_time < earliest_time:
                earliest_time = v.__last_access_time
                pop_id = v.__id
        cls._instances.pop(pop_id)

    @property
    def id(self):
        return self.__id


class JobPath(Path):

    _instances = {}
    _lock = threading.Lock()

    def __init__(self, id):
        super(JobPath, self).__init__(id)
        self.__root_dir = PathUtils.join(GflConf.data_dir, "job", id)
        self.__metadata_file = PathUtils.join(self.__root_dir, "metadata.json")
        self.__sqlite_file = PathUtils.join(self.__root_dir, "job.sqlite")
        self.__config_dir = PathUtils.join(self.__root_dir, "job")
        self.__job_config_file = PathUtils.join(self.__root_dir, "job", "job.json")
        self.__train_config_file = PathUtils.join(self.__root_dir, "job", "train.json")
        self.__aggregate_config_file = PathUtils.join(self.__root_dir, "job", "aggregate.json")
        self.__topology_config_file = PathUtils.join(self.__root_dir, "job", "topology.json")
        self.__module_name = "fl_model"
        self.__module_dir = PathUtils.join(self.__root_dir, "job")
        self.__metrics_dir = PathUtils.join(self.__root_dir, "results", "metrics")
        self.__params_dir = PathUtils.join(self.__root_dir, "results", "params")
        self.__reports_dir = PathUtils.join(self.__root_dir, "results", "reports")
        self.__client_params_dir = PathUtils.join(self.__root_dir, "round-%d", "%s", "params")
        self.__global_params_dir = PathUtils.join(self.__root_dir, "round-%d", "global", "params")
        self.__client_work_dir = PathUtils.join(self.__root_dir, "round-%d", "%s", "work")

    def makedirs(self):
        os.makedirs(self.__root_dir, exist_ok=True)
        os.makedirs(self.__config_dir, exist_ok=True)
        os.makedirs(self.__metrics_dir, exist_ok=True)
        os.makedirs(self.__params_dir, exist_ok=True)
        os.makedirs(self.__reports_dir, exist_ok=True)

    @property
    def root_dir(self):
        return self.__root_dir

    @property
    def metadata_file(self):
        return self.__metadata_file

    @property
    def sqlite_file(self):
        return self.__sqlite_file

    @property
    def config_dir(self):
        return self.__config_dir

    @property
    def job_config_file(self):
        return self.__job_config_file

    @property
    def train_config_file(self):
        return self.__train_config_file

    @property
    def aggregate_config_file(self):
        return self.__aggregate_config_file

    @property
    def topology_config_file(self):
        return self.__topology_config_file

    @property
    def module_name(self):
        return self.__module_name

    @property
    def module_dir(self):
        return self.__module_dir

    @property
    def metrics_dir(self):
        return self.__metrics_dir

    @property
    def params_dir(self):
        return self.__params_dir

    @property
    def reports_dir(self):
        return self.__reports_dir

    def client_params_dir(self, step: int, address: str):
        return self.__client_params_dir % (step, address)

    def client_work_dir(self, step: int, address: str):
        # self.__client_work_dir = PathUtils.join(self.__root_dir, "round-%d", "%s", "work")
        return self.__client_work_dir % (step, address)

    def server_params_dir(self, step: int):
        return self.client_params_dir(step, "global")

    def server_work_dir(self, step: int):
        return self.client_work_dir(step, "global")

    def global_params_dir(self, round: int):
        return self.__global_params_dir % round


class DatasetPath(Path):

    _instances = {}
    _lock = threading.Lock()

    def __init__(self, id):
        super(DatasetPath, self).__init__(id)
        self.__root_dir = PathUtils.join(GflConf.data_dir, "dataset", id)
        self.__metadata_file = PathUtils.join(self.__root_dir, "metadata.json")
        self.__config_dir = PathUtils.join(self.__root_dir, "dataset")
        self.__dataset_config_file = PathUtils.join(self.__root_dir, "dataset", "dataset.json")
        self.__module_name = "fl_dataset"
        self.__module_dir = PathUtils.join(self.__root_dir, "dataset")

    def makedirs(self):
        os.makedirs(self.__root_dir, exist_ok=True)
        os.makedirs(self.__config_dir, exist_ok=True)

    @property
    def root_dir(self):
        return self.__root_dir

    @property
    def metadata_file(self):
        return self.__metadata_file

    @property
    def config_dir(self):
        return self.__config_dir

    @property
    def dataset_config_file(self):
        return self.__dataset_config_file

    @property
    def module_name(self):
        return self.__module_name

    @property
    def module_dir(self):
        return self.__module_dir
