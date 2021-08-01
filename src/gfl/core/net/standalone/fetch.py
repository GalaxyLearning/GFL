from typing import List

from gfl.core.net.abstract import NetFetch


class StandaloneFetch(NetFetch):

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
