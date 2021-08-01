__all__ = [
    "PlainObject"
]

from typing import Any, Tuple, List, Set, Dict, _GenericAlias


"""
Supported type:
1. Fundamental type
    bool, int, float, str
2. Container type
    Tuple[T], List[T], Set[T], Dict[K, V]
3. Arbitrary type
    Any

Method does not check the type of the passed parameter and assumes that the type of the 
passed parameter can be legally converted.

Multiple inheritance is not supported.

Do not support non-standard type declarations such as Dict, tuple, bytes, etc.

There is no support for subclasses to override superclass properties with the same name.
"""


class ClassField(object):

    def __init__(self, name, annotation):
        super(ClassField, self).__init__()
        self.name = name
        self.base_type = None
        self.sub_types = None
        self.__parse_annotation(annotation)

    def __parse_annotation(self, annotation):
        if annotation in [bool, int, float, str, Any]:
            self.base_type = annotation
        elif type(annotation) == type and issubclass(annotation, PlainObject):
            self.base_type = PlainObject
            self.sub_types = [ClassMetadata(annotation)]
        else:
            self.base_type = annotation.__origin__
            self.sub_types = []
            for a in annotation.__args__:
                self.sub_types.append(ClassField(None, a))

    def encode(self, o):

        if o is None:
            return o
        if self.base_type == Any:
            return o
        if self.base_type in [bool, int, float, str]:
            if self.base_type == type(o):
                return o
            else:
                raise ValueError("%s cannot cast to %s" % (type(o), self.base_type))
        elif self.base_type == PlainObject:
            return self.sub_types[0].encode(o)
        else:
            if self.base_type == dict:
                if type(o) == dict:
                    ret = {}
                    for k, v in o.items():
                        ke = self.sub_types[0].encode(k)
                        ve = self.sub_types[1].encode(v)
                        ret[ke] = ve
                    return ret
                else:
                    raise ValueError("%s cannot cast to %s" % (type(o), self.base_type))
            else:
                if type(o) == self.base_type:
                    ret = []
                    for e in o:
                        ee = self.sub_types[0].encode(e)
                        ret.append(ee)
                    return ret
                else:
                    raise ValueError("%s cannot cast to %s" % (type(o), self.base_type))

    def decode(self, d):
        if d is None:
            return d
        if self.base_type == Any:
            return d
        if self.base_type in [bool, int, float, str]:
            if self.base_type == type(d):
                return d
            else:
                raise ValueError("%s cannot cast to %s" % (type(d), self.base_type))
        elif self.base_type == PlainObject:
            return self.sub_types[0].decode(d)
        else:
            if self.base_type == dict:
                if type(d) == dict:
                    ret = {}
                    for k, v in d.items():
                        kd = self.sub_types[0].decode(k)
                        ed = self.sub_types[1].decode(v)
                        ret[kd] = ed
                    return ret
                else:
                    raise ValueError("%s cannot cast to %s" % (type(d), self.base_type))
            else:
                if type(d) == self.base_type:
                    ret = []
                    for e in d:
                        ed = self.sub_types[0].decode(e)
                        ret.append(ed)
                    return self.base_type(ret)
                else:
                    raise ValueError("%s cannot cast to %s" % (type(d), self.base_type))


class ClassMetadata(object):

    def __init__(self, clazz):
        super(ClassMetadata, self).__init__()
        self.clazz = clazz
        self.fields = []
        self.__parse_clazz(clazz)

    def __parse_clazz(self, clazz):
        for name, at in clazz.__annotations__.items():
            self.fields.append(ClassField(name, at))

    def encode(self, o):
        ret = {}
        for f in self.fields:
            ret[f.name] = f.encode(getattr(o, f.name))
        return ret

    def decode(self, d, out_obj=None):
        if out_obj is None:
            out_obj = self.clazz()
        for f in self.fields:
            if f.name not in d:
                raise ValueError("%s field missed." % f.name)
            setattr(out_obj, f.name, f.decode(d[f.name]))
        return out_obj


metadata_cache = {}


def reflect_metadata(clazz):
    if clazz in metadata_cache:
        return metadata_cache[clazz]
    metadata = ClassMetadata(clazz)
    metadata_cache[clazz] = metadata
    return metadata


class PlainObject(object):

    def __init__(self, **kwargs):
        super(PlainObject, self).__init__()
        self._set_kwargs(type(self), **kwargs)

    def _set_kwargs(self, cls, **kwargs):
        if cls == PlainObject:
            return
        self._set_kwargs(cls.__base__, **kwargs)
        for name, _ in cls.__annotations__.items():
            if name in kwargs:
                setattr(self, name, kwargs.get(name))

    def to_dict(self):
        return self.__to_dict(type(self))

    def from_dict(self, d):
        return self.__from_dict(d, type(self))

    def __to_dict(self, cls):
        if cls == PlainObject:
            return {}
        ret = self.__to_dict(cls.__base__)
        metadata = reflect_metadata(cls)
        ret.update(metadata.encode(self))
        return ret

    def __from_dict(self, d, cls):
        if cls == PlainObject:
            return self
        self.__from_dict(d, cls.__base__)
        metadata = reflect_metadata(cls)
        metadata.decode(d, self)
        return self
