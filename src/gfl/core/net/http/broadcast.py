from typing import NoReturn

from gfl.core.net.abstract import NetBroadcast,File


class HttpBroadcast(NetBroadcast):

    @classmethod
    def broadcast_job(cls, job_id: str, job: File) -> NoReturn:
        pass

    @classmethod
    def broadcast_dataset(cls, dataset_id: str, dataset: File) -> NoReturn:
        pass

    @classmethod
    def broadcast_global_params(cls, job_id: str, step: int, params: File) -> NoReturn:
        pass
