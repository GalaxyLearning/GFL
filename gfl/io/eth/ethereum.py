from web3 import Web3

from gfl.io.eth.contract import Contract


class Ethereum(object):

    def __init__(self, url):
        super(Ethereum, self).__init__()
        self.__w3 = Web3(Web3.HTTPProvider(url))

        self.coinbase = self.__w3.eth.coinbase
        self.gas_price = self.__w3.eth.gasPrice
        self.new_account = self.__w3.geth.personal.newAccount
        self.unlock_account = self.__w3.geth.personal.unlockAccount
        self.lock_account = self.__w3.geth.personal.lockAccount

        self.estimate_gas = self.__w3.eth.estimateGas
        self.get_transaction_count = self.__w3.eth.getTransactionCount

        self.send_transaction = self.__w3.eth.sendTransaction
        self.send_raw_transaction = self.__w3.eth.sendRawTransaction
        self.wait_for_transaction_receipt = self.__w3.eth.waitForTransactionReceipt

    def contract(self, name, address=None):
        return Contract(name=name, address=address, w3=self.__w3)

    def controller_contract(self, address=None):
        return self.contract("Conr", address)

