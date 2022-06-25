from typing import Tuple

from gfl.core.net.abstract import NetReceive, File


class HttpReceive(NetReceive):

    @classmethod
    def receive_job(cls) -> Tuple:
        pass

    @classmethod
    def receive_partial_params(cls, client: str, job_id: str, step: int) -> File:
        pass
