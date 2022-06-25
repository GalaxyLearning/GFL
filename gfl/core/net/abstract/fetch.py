from typing import List

from gfl.core.lfs.types import File


"""
fetch
    job:
        fetch_job(job_id: str) -> ServerJob
        list_job_id() -> List[str]

    dataset:
        fetch_dataset(dataset_id: str) -> Dataset
        list_dataset_id() -> List[str]

    params:
    
"""


class NetFetch(object):

    @classmethod
    def fetch_job(cls, job_id: str):
        pass

    @classmethod
    def list_job_id(cls) -> List[str]:
        pass

    @classmethod
    def fetch_dataset(cls, dataset_id: str):
        pass

    @classmethod
    def list_dataset_id(cls) -> List[str]:
        pass
