from typing import NoReturn

from gfl.core.lfs.types import File


"""
send

    job:
        

    dataset:
        

    params:
        send_partial_params(client: str, job_id: str, step: int, params: File) -> NoReturn

"""


class NetSend(object):

    @classmethod
    def send_partial_params(cls, client: str, job_id: str, step: int, params: File) -> NoReturn:
        pass
