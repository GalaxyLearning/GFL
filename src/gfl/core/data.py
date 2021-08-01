__all__ = [
    "DatasetMetadata",
    "JobMetadata",
    "Dataset",
    "Job"
]

from typing import Union

from gfl.core.config import DatasetConfig, TrainConfig, AggregateConfig, JobConfig
from gfl.utils.po_utils import PlainObject


class Metadata(PlainObject):
    id: str = None
    owner: str = None
    create_time: int
    content: str


class DatasetMetadata(Metadata):
    pass


class JobMetadata(Metadata):
    pass


class Dataset(object):

    def __init__(self, *,
                 dataset_id: str = None,
                 metadata: DatasetMetadata = None,
                 dataset_config: DatasetConfig = None):
        super(Dataset, self).__init__()
        self.module = None
        self.dataset_id = dataset_id
        self.metadata = metadata if isinstance(metadata, DatasetMetadata) else DatasetMetadata().from_dict(metadata)
        self.dataset_config = dataset_config if isinstance(dataset_config, DatasetConfig) \
            else DatasetConfig().from_dict(dataset_config)


class Job(object):

    def __init__(self, *,
                 job_id: str = None,
                 metadata: Union[JobMetadata, dict] = None,
                 job_config: Union[JobConfig, dict] = None,
                 train_config: Union[TrainConfig, dict] = None,
                 aggregate_config: Union[AggregateConfig, dict] = None):
        super(Job, self).__init__()
        self.module = None
        self.job_id = job_id
        self.cur_round = 0          # Deprecated
        self.metadata = metadata if isinstance(metadata, JobMetadata) else JobMetadata().from_dict(metadata)
        self.job_config = job_config if isinstance(job_config, JobConfig) else JobConfig().from_dict(job_config)
        self.train_config = train_config if isinstance(train_config, TrainConfig) \
            else TrainConfig().from_dict(train_config)
        self.aggregate_config = aggregate_config if isinstance(aggregate_config, AggregateConfig) \
            else AggregateConfig().from_dict(aggregate_config)
        self.dataset = None

    def mount_dataset(self, dataset: Dataset):
        self.dataset = dataset
