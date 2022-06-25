

from .config import GflConfig


class GflApplication(object):

    def __init__(self, home, config: GflConfig):
        super(GflApplication, self).__init__()
        self.home = home
        self.config = config

    def init(self):
        pass

    def start(self, daemon=False):
        pass
