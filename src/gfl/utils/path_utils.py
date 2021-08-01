import os
from pathlib import PurePath


class PathUtils(object):

    @classmethod
    def join(cls, *paths):
        return PurePath(*paths).as_posix()

    @classmethod
    def user_home_dir(cls):
        return cls.join(os.path.expanduser("~"))

    @classmethod
    def src_root_dir(cls):
        return cls.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    @classmethod
    def exists(cls, path):
        return os.path.exists(path)

    @classmethod
    def abspath(cls, path):
        return cls.join(os.path.abspath(path))
