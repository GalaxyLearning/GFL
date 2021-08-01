__all__ = [
    "DatasetGenerator",
    "JobGenerator"
]

import abc
import time
import uuid

from gfl.core.manager.node import GflNode
from gfl.core.config import *
from gfl.core.data import JobMetadata, DatasetMetadata, Job, Dataset
from gfl.utils import TimeUtils


class Generator(object):

    def __init__(self, module):
        super(Generator, self).__init__()
        self.module = module

    @abc.abstractmethod
    def generate(self):
        pass

    @classmethod
    def _generate_job_id(cls):
        return uuid.uuid4().hex

    @classmethod
    def _generate_dataset_id(cls):
        return uuid.uuid4().hex


class DatasetGenerator(Generator):

    def __init__(self, module):
        super(DatasetGenerator, self).__init__(module)
        self.dataset_id = self._generate_dataset_id()
        self.metadata = DatasetMetadata(id=self.dataset_id,
                                        owner=GflNode.address,
                                        create_time=TimeUtils.millis_time())
        self.dataset_config = DatasetConfig(module=module)

    def generate(self):
        dataset = Dataset(dataset_id=self.dataset_id,
                          metadata=self.metadata,
                          dataset_config=self.dataset_config)
        dataset.module = self.module
        return dataset


class JobGenerator(Generator):

    def __init__(self, module):
        super(JobGenerator, self).__init__(module)
        self.job_id = self._generate_job_id()
        self.metadata = JobMetadata(id=self.job_id,
                                    owner=GflNode.address,
                                    create_time=TimeUtils.millis_time())
        self.job_config = JobConfig(module=module)
        self.train_config = TrainConfig(module=module)
        self.aggregate_config = AggregateConfig(module=module)

    def generate(self):
        job = Job(job_id=self.job_id,
                  metadata=self.metadata,
                  job_config=self.job_config,
                  train_config=self.train_config,
                  aggregate_config=self.aggregate_config)
        job.module = self.module
        return job
