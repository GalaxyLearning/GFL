
import zcommons.time as time

from .data import *
from .ipc_adapter import IpcAdapter


class GFL(object):

    def __init__(self, ipc_adapter: IpcAdapter):
        super(GFL, self).__init__()
        self.__ipc_adapter = ipc_adapter

    def generate_job(self,
                     content: str,
                     job_config: JobConfig,
                     train_config: TrainConfig,
                     aggregate_config: AggregateConfig,
                     datasets: list,
                     module):
        meta = JobMeta(content=content, datasets=datasets)
        job = Job(meta=meta,
                  job_config=job_config,
                  train_config=train_config,
                  aggregate_config=aggregate_config)
        job = JobMeta(id="", owner="", create_time=time.milli_time(), content=content,
                      job_config=job_config, train_config=train_config, aggregate_config=aggregate_config)

    def generate_dataset(self,
                         content: str,
                         dataset_config: DatasetConfig):
        dataset = DatasetMeta(id="", owner="", create_time=time.milli_time(), content=content,
                              dataset_config=dataset_config)

    def __pack_job(self, job):
        pass