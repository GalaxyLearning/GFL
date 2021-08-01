from typing import Tuple

from gfl.core.lfs.types import File

"""
receive

    job:
        receive_job() -> List[str]

    dataset:

    params:
        receive_partial_params(client: str, job_id: str, step: int) -> File

"""


class NetReceive(object):

    @classmethod
    def receive_job(cls) -> Tuple:
        pass

    @classmethod
    def receive_partial_params(cls, client: str, job_id: str, step: int) -> File:
        pass
