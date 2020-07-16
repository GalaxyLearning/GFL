# Copyright (c) 2020 GalaxyLearning Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.




import json
import types

from typing import Union

class JsonUtil(object):

    __primary_types = [
        type(None), bool, int, float, str
    ]
    __container_types = [
        tuple, list, set, dict
    ]
    __builtin_types = []
    __builtin_types.extend(__primary_types)
    __builtin_types.extend(__container_types)

    @classmethod
    def is_primary_type(cls, tp):
        return tp in JsonUtil.__primary_types

    @classmethod
    def is_container_type(cls, tp):
        return tp in JsonUtil.__container_types

    @classmethod
    def is_builtin_type(cls, tp):
        return tp in JsonUtil.__builtin_types

    @classmethod
    def to_json(cls, obj):
        return json.dumps(cls.to_json_obj(obj), ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str, obj_type, encoding="utf-8"):
        return cls.from_json_obj(json.loads(json_str, encoding=encoding), obj_type)

    @classmethod
    def to_json_obj(cls, obj):
        tp = type(obj)
        if cls.is_primary_type(tp):
            return obj
        if tp in [list, tuple, set]:
            data = []
            for e in obj:
                data.append(cls.to_json_obj(e))
            return data
        if tp in [dict]:
            data = {}
            for k, v in obj.items():
                data[str(k)] = cls.to_json_obj(v)
            return data
        attrs = cls._get_attrs(tp)
        data = {}
        for k in attrs.keys():
            data[k] = cls.to_json_obj(getattr(obj, k, None))
        return data

    @classmethod
    def from_json_obj(cls, json_obj, target_type: Union[type, list, tuple]):
        if json_obj is None:
            return None
        json_type = type(json_obj)
        if type(target_type) not in [list, tuple]:
            target_type = (target_type, )
        if cls.is_container_type(target_type[0]):
            if target_type[0] == dict:
                if json_type != dict or len(target_type) < 3:
                    raise TypeError(cls.__splice_type_error(json_type, target_type[0]))
                ret_data = {}
                for k, v in json_obj.items():
                    ret_data[JsonUtil.from_json_obj(k, target_type[1])] = JsonUtil.from_json_obj(v, target_type[2])
                return ret_data
            if target_type[0] == set:
                if json_type != list or len(target_type) < 2:
                    raise TypeError(cls.__splice_type_error(json_type, target_type[0]))
                ret_data = set()
                for e in json_obj:
                    ret_data.add(cls.from_json_obj(e, target_type[1]))
                return ret_data
            if target_type[0] == tuple or target_type[0] == list:
                if json_type != list or len(target_type) < 2:
                    raise TypeError(cls.__splice_type_error(json_type, target_type[0]))
                ret_data = []
                for e in json_obj:
                    ret_data.append(cls.from_json_obj(e, target_type[1]))
                if target_type[0] == tuple:
                    return tuple(ret_data)
                else:
                    return ret_data
        else:
            if cls.is_primary_type(target_type[0]):
                if json_type != target_type[0]:
                    raise TypeError(cls.__splice_type_error(json_type, target_type[0]))
                return json_obj
            attrs = cls._get_attrs(target_type[0])
            ret_data = target_type[0]()
            for k, v in attrs.items():
                setattr(ret_data, k, cls.from_json_obj(json_obj.get(k), v))
            return ret_data

    @classmethod
    def __splice_type_error(cls, left, right):
        return "%s cannot convert to %s." % (left, right)

    @classmethod
    def _get_attrs(cls, tp):
        attrs = {}
        if JsonUtil.is_builtin_type(tp):
            return attrs
        for k, v in tp.__dict__.items():
            if k.startswith("_"):
                continue
            if isinstance(v, types.FunctionType) or isinstance(v, classmethod) or isinstance(v, staticmethod):
                continue
            attrs[k] = v
        return attrs