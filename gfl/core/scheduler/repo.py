__all__ = [
    "KVEntity",
    "ClientEntity",
    "ParamsEntity",
    "KVRepo",
    "ClientRepo",
    "ParamsRepo"
]

import abc
import threading
import time
from typing import Tuple, List

from gfl.core.context import SqliteContext
from gfl.core.lfs import JobPath
from gfl.utils import PlainObject


class KVEntity(PlainObject):

    key: str = None
    value: str = None


class ClientEntity(PlainObject):

    address: str = None
    dataset: str = None
    pub_key: str = None


class ParamsEntity(PlainObject):

    id: str = None
    step: str = None
    is_global: bool = None
    node_address: str = None
    params: str = None
    params_check: str = None


class Repo(object):

    _instances = {}
    _lock = threading.Lock()

    def __init__(self, job_id):
        super(Repo, self).__init__()
        self.__job_id = job_id
        self.__last_access_time = time.time()
        if not hasattr(self, "has_inited"):
            self.has_inited = False
        if not self.has_inited:
            self.init_table()
            self.has_inited = True

    def __new__(cls, job_id):
        if job_id is None:
            return object.__new__(cls)
        cls._lock.acquire()
        if len(cls._instances) > 127:
            cls.__clear()
        obj = cls._instances.get(job_id)
        if obj is None:
            obj = object.__new__(cls)
            cls._instances[job_id] = obj
        cls._lock.release()
        return obj

    @classmethod
    def __clear(cls):
        pop_id = ""
        earliest_time = time.time()
        for k, v in cls._instances.items():
            if v.__last_access_time < earliest_time:
                earliest_time = v.__last_access_time
                pop_id = v.__id
        cls._instances.pop(pop_id)

    @property
    def job_id(self):
        return self.__job_id

    @classmethod
    @abc.abstractmethod
    def table_info(cls) -> Tuple[str, List[str], List[str]]:
        raise NotImplementedError("")

    def execute_update(self, statements: List[str], *params):
        job_path = JobPath(self.job_id)
        with SqliteContext(job_path.sqlite_file) as (_, cursor):
            for st in statements:
                cursor.execute(st, tuple(params))

    def execute_query(self, statements: List[str], *params):
        job_path = JobPath(self.job_id)
        with SqliteContext(job_path.sqlite_file) as (_, cursor):
            for st in statements:
                cursor.execute(st, tuple(params))
            ret = cursor.fetchall()
        return ret

    def init_table(self):
        name, _, statements = self.table_info()
        job_path = JobPath(self.job_id)
        with SqliteContext(job_path.sqlite_file) as (_, cursor):
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            table_names = [r[0] for r in cursor.fetchall()]
            if name not in table_names:
                for st in statements:
                    cursor.execute(st)


class KVRepo(Repo):

    _instances = {}

    def table_info(cls) -> Tuple[str, List[str], List[str]]:
        return "kv", ["key", "value"], [
            "create table kv (key text not null, value text);",
            "create unique index kv_key_uindex on kv (key);"
        ]

    def save(self, kv: KVEntity):
        self.execute_update(["INSERT INTO kv(key, value) VALUES (?, ?)"], kv.key, kv.value)

    def get(self, key: str):
        entities = self.execute_query(["SELECT * FROM kv WHERE key=?"], key)
        if len(entities) != 1:
            raise ValueError("")
        return KVEntity(key=entities[0][0], value=entities[0][1])

    def update(self, kv: KVEntity):
        self.execute_update(["UPDATE kv SET value=? WHERE key=?"], kv.value, kv.key)


class ClientRepo(Repo):

    _instances = {}

    def table_info(cls) -> Tuple[str, List[str], List[str]]:
        return "client", ["address", "dataset", "pub_key"], [
            "create table client (address text not null, dataset text not null, pub_key text not null);",
            "create unique index client_address_uindex on client (address);"
        ]

    def save(self, client: ClientEntity):
        self.execute_update(["INSERT INTO client(address, dataset, pub_key) VALUES (?, ?, ?)"], client.address, client.dataset, client.pub_key)

    def get(self, address: str):
        entities = self.execute_query(["SELECT address, dataset, pub_key FROM client WHERE address=?"], address)
        if len(entities) != 1:
            raise ValueError("")
        entity = entities[0]
        return ClientEntity(address=entity[0], dataset=entity[1], pub_key=entity[2])

    def get_all(self):
        entities = self.execute_query(["SELECT address, dataset, pub_key FROM client"])
        ret = []
        for entity in entities:
            ret.append(ClientEntity(address=entity[0], dataset=entity[1], pub_key=entity[2]))
        return ret


class ParamsRepo(Repo):

    _instances = {}

    def table_info(cls) -> Tuple[str, List[str], List[str]]:
        return "params", ["id", "step", "is_global", "node_address", "params", "params_check"], [
            "create table params (id integer not null constraint params_pk primary key autoincrement,"
            "step integer not null, is_global integer default 0 not null, node_address text not null, "
            "params text not null, params_check text not null);"
        ]

    def save(self, params: ParamsEntity):
        self.execute_update(["INSERT INTO params(step, is_global, node_address, params, params_check) VALUES (?, ?, ?, ?, ?)"],
                            params.step, params.is_global, params.node_address, params.params, params.params_check)

    def get(self, node_address: str, step: int):
        entities = self.execute_query(["SELECT id, step, is_global, node_address, params, params_check FROM params " \
                                        "WHERE step=? AND node_address=?"], step, node_address)
        if len(entities) != 1:
            raise ValueError("")
        entity = entities[0]
        return ParamsEntity(id=entity[0], step=entity[1], is_global=entity[2], node_address=entity[3], params=entity[4], params_check=entity[5])

    def get_all(self):
        entities = self.execute_query(["SELECT id, step, is_global, node_address, params, params_check FROM params"])
        ret = []
        for entity in entities:
            ret.append(ParamsEntity(id=entity[0], step=entity[1], is_global=entity[2], node_address=entity[3], params=entity[4], params_check=entity[5]))
        return ret
