import os

from .path import Path


class Lfs(object):

    def __init__(self, home):
        super(Lfs, self).__init__()
        self.__path = Path(home)

    @property
    def home(self):
        return str(self.__path.home())

    @property
    def path(self):
        return self.__path

    def init(self, overwrite=False):
        if not self.__path.home().exists():
            self.__path.home().makedirs()
        if len(os.listdir(self.__path.home())) > 0:
            if overwrite:
                self.__path.home().rm()
                self.__path.home().makedirs()
            else:
                raise ValueError(f"home dir{self.home} is exists and not empty")
        self.__path.data_dir().makedirs()
        self.__path.logs_dir().makedirs()
        self.__path.job.root_dir().makedirs()
        self.__path.dataset.root_dir().makedirs()
