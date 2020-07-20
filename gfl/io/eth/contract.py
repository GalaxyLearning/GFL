from typing import Union, Any, NoReturn

from web3 import Web3
from web3.eth import Address, ChecksumAddress

from gfl.io.eth.abi import load_abi, load_bytecode, load_address
from gfl.io.eth.exception import TransactionTimeoutException


EthAddress = Union[str, Address, ChecksumAddress]


class Contract(object):

    def __init__(self,
                 w3: Web3,
                 name: str,
                 address: EthAddress = None,
                 abi = None,
                 bytecode: str = None):
        super(Contract, self).__init__()
        self.name = name

        self.address = address if address is not None else load_address(name)
        self.abi = abi if abi is not None else load_abi(name)
        self.bytecode = bytecode if bytecode is not None else load_bytecode(name)

        self._w3 = w3
        self.__web3_contract = self._w3.eth.contract(address=address, abi=self.abi, bytecode=self.bytecode)

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

    def transact(self, name, args: Union[tuple, list] = (), transaction: dict = None) -> str:
        """
        Execute the specified method by sending a new public transaction.

        :param name:
        :param args:
        :param transaction:
        :return:
        """
        method = self.__web3_contract.get_function_by_name(name)
        return method(*args).transact(transaction=transaction).hex()

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
        method = self.__web3_contract.get_function_by_name(name)
        tx_hash = method(*args).transact(transaction=transaction)
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

    def __init__(self,
                 w3: Web3,
                 address: EthAddress = None,
                 abi = None,
                 bytecode: str = None):
        super(ControllerContract, self).__init__(w3=w3, name="controller", address=address,
                                                 abi=abi, bytecode=bytecode)

    def set_job_mapping_storage_contract(self, job_id: str, storage_contract_address: EthAddress) -> dict:
        """

        :param job_id:
        :param storage_contract_address:
        :return:
        """
        return self.sync_transact("setJobMappingStorageContract", (job_id, storage_contract_address))

    def get_job_mapping_storage_contract(self, job_id: str) -> str:
        return self.call("getJobMappingStorageContract", (job_id, ))

    def set_job_status(self, job_id: str, status: int) -> dict:
        """

        :param job_id:
        :param status: 0->prepared, 1->running, 2->finished
        :return:
        """
        return self.sync_transact("setJobStatus", (job_id, status))

    def get_job_status(self, job_id: str) -> int:
        return self.call("getJobStatus", (job_id, ))


class StorageContract(Contract):

    def __init__(self,
                 w3: Web3,
                 address: EthAddress = None,
                 abi = None,
                 bytecode: str = None):
        super(StorageContract, self).__init__(w3=w3, name="storage", address=address,
                                                 abi=abi, bytecode=bytecode)

    def set_fed_step(self, fed_step: int) -> dict:
        return self.sync_transact("setFedStep", (fed_step, ))

    def get_fed_step(self) -> int:
        return self.call("getFedStep")

    def update_activate_clients(self, client_address: EthAddress, op: int) -> dict:
        """

        :param client_id:
        :param op: 0->join, 1->exit
        :return:
        """
        return self.sync_transact("updateActivateClients", (client_address, op))

    def upload_model_parameters_ipfs_hash(self, fed_step: int,
                                          client_address: EthAddress,
                                          model_pars_ipfs_hash: str) -> dict:
        return self.sync_transact("uploadModelParametersIpfsHash", (fed_step, client_address, model_pars_ipfs_hash))

    # Has no impl in contract
    def selection_client_address(self, fed_step: int) -> Any:
        return self.sync_transact("selectionClientAddress", [fed_step])

    # Has no impl in contract
    def get_selection_client_address(self, fed_step) -> Address:
        return self.call("getSelectionClientAddress", [fed_step])

    def download_model_parameters_ipfs_hash(self, fed_step: int, client_address: EthAddress) -> tuple:
        return self.call("downloadModelParametersIpfsHash", (fed_step, client_address))

    def upload_aggregated_parameters_ipfs_hash(self, fed_step: int, model_client_address: EthAddress, aggregated_model_pars_ipfs_hash: str) -> dict:
        return self.sync_transact("uploadAggreagtedParametersIpfsHash", (fed_step, model_client_address, aggregated_model_pars_ipfs_hash))

    def download_aggregated_parameters_ipfs_hash(self, fed_step: int) -> tuple:
        return self.call("downloadAggregatedParametersIpfsHash", (fed_step, ))

    def download_final_aggregated_parameters_ipfs_hash(self) -> str:
        return self.call("downloadFinalAggregatedParametersIpfsHash")

    def upload_final_aggregated_parameters_ipfs_hash(self, final_aggregated_ipfs_hash: str) -> dict:
        return self.sync_transact("uploadFinalAggregatedParametersIpfsHash", (final_aggregated_ipfs_hash, ))







