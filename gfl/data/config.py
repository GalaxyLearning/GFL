__all__ = [
    "ObjectConfig",
    "JobConfig",
    "TrainConfig",
    "AggregateConfig",
    "DatasetConfig"
]

from dataclasses import dataclass
from typing import Dict, Any


@dataclass()
class ObjectConfig:

    name: str
    is_builtin: bool = False
    is_instance: bool = False
    args: Dict[str, Any] = None


@dataclass()
class JobConfig:

    trainer: ObjectConfig
    aggregator: ObjectConfig


@dataclass()
class TrainConfig:

    model: ObjectConfig
    optimizer: ObjectConfig
    criterion: ObjectConfig
    lr_scheduler: ObjectConfig = None
    epoch: int = 10
    batch_size: int = 32


@dataclass()
class AggregateConfig:

    global_epoch: int = 50


@dataclass()
class DatasetConfig:

    dataset: ObjectConfig
    val_dataset: ObjectConfig = None
    val_rate: float = 0.2
