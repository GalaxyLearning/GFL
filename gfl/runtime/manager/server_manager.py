

class ServerManager(object):

    def __init__(self, fs, node, config):
        super(ServerManager, self).__init__()
        self.__fs = fs
        self.__node = node
        self.__config = config
