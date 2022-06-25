__all__ = [
    "Job",
    "Dataset"
]

from dataclasses import dataclass

from .config import *
from .meta import *


@dataclass()
class Job:

    meta: JobMeta
    job_config: JobConfig
    train_config: TrainConfig
    aggregate_config: AggregateConfig


@dataclass()
class Dataset:

    meta: DatasetMeta
    dataset_config: DatasetConfig
