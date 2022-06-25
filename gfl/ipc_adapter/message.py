import base64
import json
from dataclasses import dataclass
from typing import Dict

import zcommons as zc


@dataclass
class Message:

    cmd: str
    args: Dict[str, str] = None
    attrs: Dict[str, str] = None
    content: str = None


class _JsonInterface(object):

    def to_json_str(self, pretty=False):
        data_dict = zc.o2d.asdict(self)
        if "files" in data_dict and data_dict["files"] is not None:
            files = {}
            for name, data in data_dict["files"].items():
                files[name] = base64.encodebytes(data).decode("ascii").strip()
            data_dict["files"] = files
        if pretty:
            return json.dumps(data_dict, indent=4)
        else:
            return json.dumps(data_dict)

    @classmethod
    def from_json_str(cls, json_str):
        if json_str is None:
            return None
        data_dict = json.loads(json_str)
        if "files" in data_dict and data_dict["files"] is not None:
            files = {}
            for name, data in data_dict["files"].items():
                files[name] = base64.decodebytes(data.encode("ascii"))
            data_dict["files"] = files
        return zc.o2d.asobj(cls, data_dict)


@dataclass()
class Request(_JsonInterface):

    cmd: str
    args: Dict[str, str] = None
    attrs: Dict[str, str] = None
    content: str = None
    files: Dict[str, bytes] = None


@dataclass()
class Response(_JsonInterface):

    status: int
    attrs: Dict[str, str] = None
    content: str = None
    files: Dict[str, bytes] = None
