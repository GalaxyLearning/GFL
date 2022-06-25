import abc
import base64
import json
import threading

import zcommons as zc

from .message import Request, Response


def _base64_encode(data: bytes):
    return base64.encodebytes(data).decode("ascii").strip()


def _base64_decode(s: str):
    return base64.decodebytes(s.encode("ascii"))


class IpcAdapter(object):

    def __init__(self, home=None, ip=None, port=None):
        super(IpcAdapter, self).__init__()
        self._home = home
        self._ip = ip
        self._port = port
        self._alive = False

    def send(self, req: Request):
        return self._send(req.to_json_str())

    def recv(self, block: bool = True) -> Response:
        data = self._recv(block)
        return Response.from_json_str(data)

    def close(self):
        self._alive = False
        self._close()

    @abc.abstractmethod
    def _send(self, data: str) -> bool:
        pass

    @abc.abstractmethod
    def _recv(self, block: bool) -> str:
        pass

    @abc.abstractmethod
    def _close(self):
        pass


class IpcServerAdapter(object):

    def __init__(self, path=None, ip=None, port=None, handler=None):
        super(IpcServerAdapter, self).__init__()
        self._path = path
        self._ip = ip
        self._port = port
        self._handler = handler
        self._alive = False

    def listen(self, back: bool = True):
        if back:
            threading.Thread(target=self._listen).start()
        else:
            self._listen()

    def close(self):
        self._alive = False
        self._close()

    @abc.abstractmethod
    def _listen(self):
        pass

    @abc.abstractmethod
    def _close(self):
        pass
