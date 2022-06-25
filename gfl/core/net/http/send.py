from typing import NoReturn

from gfl.core.net.abstract import NetSend, File


class HttpSend(NetSend):

    @classmethod
    def send_partial_params(cls, client: str, job_id: str, step: int, params: File) -> NoReturn:
        pass

