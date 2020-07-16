
class IpfsException(Exception):

    def __init__(self):
        super(IpfsException, self).__init__()


class IpfsInvalidHashException(IpfsException):

    def __init__(self):
        super(IpfsInvalidHashException, self).__init__()


