from typing import NoReturn

from gfl.core.lfs.types import File
"""
broadcast

    job:
        broadcast_job(job_id: str, job: File) -> NoReturn

    dataset:
        broadcast_dataset(dataset_id: str, dataset: File) -> NoReturn

    params:
        broadcast_global_params(job_id: str, step: int, params: File) -> NoReturn
      
"""


class NetBroadcast(object):

    @classmethod
    def broadcast_job(cls, job_id: str, job: File) -> NoReturn:
        pass

    @classmethod
    def broadcast_dataset(cls, dataset_id: str, dataset: File) -> NoReturn:
        pass

    @classmethod
    def broadcast_global_params(cls, job_id: str, step: int, params: File) -> NoReturn:
        pass
