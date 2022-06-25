import socket
import threading
import typing

import zcommons as zc

from .ipc import IpcAdapter, IpcServerAdapter
from .message import Request, Response


class SocketHandler(object):

    def __init__(self, conn: socket.socket, addr=None):
        super(SocketHandler, self).__init__()
        self.__conn = conn
        self.__addr = addr

    def send(self, data: bytes) -> bool:
        data_len = len(data)
        byte_len = data_len.to_bytes(4, "big")
        sent_len = 0
        while sent_len < 4:
            sent_len += self.__conn.send(byte_len[sent_len:])
        sent_len = 0
        while sent_len < data_len:
            sent_len += self.__conn.send(data[sent_len:])
        return True

    def recv(self, block) -> bytes:
        data = None
        if not block:
            self.__conn.setblocking(False)
        try:
            byte_len = self.__conn.recv(4)
            data_len = int.from_bytes(byte_len, "big")
            data_list = []
            recv_len = 0
            while recv_len < data_len:
                d = self.__conn.recv(data_len - recv_len)
                recv_len += len(d)
                data_list.append(d)
            data = bytes().join(data_list)
        except:
            pass
        if not block:
            self.__conn.setblocking(True)
        return data

    @property
    def addr(self):
        return self.__addr


class SocketAdapter(IpcAdapter):

    def __init__(self, ip=None, port=None):
        super(SocketAdapter, self).__init__(None, ip, port)
        self.__client_socket = socket.socket()
        self.__client_socket.connect((ip, port))
        self.__handler = SocketHandler(self.__client_socket, None)

    def _send(self, data: str) -> bool:
        return self.__handler.send(data.encode("utf8"))

    def _recv(self, block: bool) -> typing.Union[str, None]:
        data = self.__handler.recv(block)
        if data is None:
            return data
        return data.decode("utf8")

    def _close(self):
        self.__client_socket.close()


class SocketServerAdapter(IpcServerAdapter):

    def __init__(self, bind_ip="127.0.0.1", bind_port=10701, handler=None):
        super(SocketServerAdapter, self).__init__(ip=bind_ip, port=bind_port, handler=handler)
        self.__server_socket = None
        self.__client_handlers = {}
        self.__alive_counter = zc.MultiThreadCounter(0, 1)

    @property
    def client_count(self):
        return self.__alive_counter.count()

    def _listen(self):
        self.__client_handlers = []
        self.__server_socket = socket.socket()
        self.__server_socket.bind((self._ip, self._port))
        self.__server_socket.listen()
        self._alive = True

        while self._alive:
            conn, addr = self.__server_socket.accept()
            t = threading.Thread(target=self.__handle, args=(conn, addr))
            t.start()

    def __handle(self, conn, addr):
        self.__alive_counter.inc()
        client_id = f"{addr[0]}:{addr[1]}"

        try:
            handler = SocketHandler(conn, addr)
            self.__client_handlers[client_id] = handler
            while True:
                data = handler.recv(True)
                data_str = data.decode("utf8")
                req: Request = Request.from_json_str(data_str)
                resp: Response = self._handler(req)
                handler.send(resp.to_json_str().encode("utf8"))
                if req.cmd == "close":
                    break
                if req.cmd == "close_server":
                    threading.Thread(target=self._close).start()
                    break
        except:
            pass

        if client_id in self.__client_handlers:
            del self.__client_handlers[client_id]
        if conn is not None:
            conn.close()
        self.__alive_counter.dec()

    def _close(self):
        self.__server_socket.close()
        client = SocketAdapter(ip=self._ip, port=self._port)
        client.send(Request("close"))
        client.close()
