
from gfl.data import JobMeta, DatasetMeta


class Generator(object):

    def __init__(self, module):
        super(Generator, self).__init__()
        self.module = module


class JobGenerator(Generator):

    def __init__(self, module):
        super(JobGenerator, self).__init__(module)

    def generate(self):
        pass


class DatasetGenerator(Generator):

    def __init__(self, module):
        super(DatasetGenerator, self).__init__(module)

    def generate(self):
        pass
