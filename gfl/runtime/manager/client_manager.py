

class ClientManager(object):

    def __init__(self, fs, node, config, client):
        super(ClientManager, self).__init__()
        self.__fs = fs
        self.__node = node
        self.__config = config
        self.__client = client

    def startup(self):
        pass
