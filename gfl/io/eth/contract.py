from typing import Union, Any

from web3 import Web3
from web3.eth import Address, ChecksumAddress

from gfl.io.eth.abi import load_abi, load_bytecode
from gfl.io.eth.exception import TransactionTimeoutException


class Contract(object):

    def __init__(self,
                 w3: Web3,
                 name: str,
                 address: Union[str, Address, ChecksumAddress],
                 abi = None,
                 bytecode: str = None):
        super(Contract, self).__init__()
        self.name = name
        self.address = address

        self.abi = abi if abi is not None else load_abi(name)
        self.bytecode = bytecode if bytecode is not None else load_bytecode(name)

        self._w3 = w3
        self.__web3_contract = self._w3.eth.contract(address, abi=self.abi, bytecode=self.bytecode)

    def call(self, name: str, args: Union[tuple, list] = (), transaction: dict = None, block_identifier = "latest") -> Any:
        """
        Call an contract method, executing the transaction locally.

        :param name:
        :param args:
        :param transaction:
        :param block_identifier:
        :return:
        """
        method = self.__web3_contract.get_function_by_name(name)
        return method(*args).call(transaction=transaction, block_identifier=block_identifier)

    def transact(self, name, args: Union[tuple, list] = (), transaction: dict = None) -> dict:
        """
        Execute the specified method by sending a new public transaction.

        :param name:
        :param args:
        :param transaction:
        :return:
        """
        method = self.__web3_contract.get_function_by_name(name)
        return method(*args).transact(transaction=transaction)

    def sync_transact(self, name, args: Union[tuple, list] = (), transaction: dict = None, timeout: int = 120) -> dict:
        """
         Execute the specified method and wait for the transaction to be included in a block.

        :param method_name: method name.
        :param name:
        :param args:
        :param transaction:
        :param timeout:
        :return:
        """
        tx_hash = self.transact(name, args, transaction)
        try:
            return self._w3.eth.waitForTransactionReceipt(tx_hash, timeout=timeout)
        except:
            raise TransactionTimeoutException(tx_hash)

    def deploy(self, constructor_args=()) -> str:
        """

        :param constructor_args:
        :return:
        """
        contract = self._w3.eth.contract(abi=self.abi, bytecode=self.bytecode)
        tx_hash = contract.constructor(*constructor_args).transact()
        try:
            tx_receipt = self._w3.eth.waitForTransactionReceipt(tx_hash)
            self.address = tx_receipt["contractAddress"]
            return self.address
        except:
            raise TransactionTimeoutException(tx_hash)


class ControllerContract(Contract):

    def __init__(self):
        super(ControllerContract, self).__init__()

    def controll(self, args=()): # Job
        self.call("controll", ())


class StorageContract(Contract):

    def __init__(self):
        super(StorageContract, self).__init__()