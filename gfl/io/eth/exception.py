
class EthereumException(Exception):

    def __init__(self, msg):
        super(EthereumException, self).__init__(msg)
        self.msg = msg

    def get_message(self):
        return self.msg


class TransactionTimeoutException(EthereumException):

    def __init__(self, hash):
        super(TransactionTimeoutException, self).__init__("Transaction receipt timeout: " + hash)


