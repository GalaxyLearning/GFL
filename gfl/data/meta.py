__all__ = [
    "JobMeta",
    "DatasetMeta"
]

from dataclasses import dataclass
from typing import List


@dataclass()
class Metadata:

    id: str = None
    owner: str = None
    create_time: int = None
    content: str = None


@dataclass()
class JobMeta(Metadata):

    datasets: List[str] = None


@dataclass()
class DatasetMeta(Metadata):

    size: int = 0
    used_cnt: int = 0
