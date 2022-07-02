__all__ = [
    "JobConfig",
    "TrainConfig",
    "AggregateConfig",
    "DatasetConfig"
]

from dataclasses import dataclass


@dataclass()
class JobConfig:

    trainer: str
    aggregator: str


@dataclass()
class TrainConfig:

    model: str
    optimizer: str
    criterion: str
    lr_scheduler: str = None
    epoch: int = 10
    batch_size: int = 32


@dataclass()
class AggregateConfig:

    global_epoch: int = 50


@dataclass()
class DatasetConfig:

    dataset: str
    val_dataset: str = None
    val_rate: float = 0.2
