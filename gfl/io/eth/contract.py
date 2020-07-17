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

    def set_job_mapping_storage_contract(self, job_id: str, storage_contract_address: Address) -> Any:
        """

        :param job_id:
        :param storage_contract_address:
        :return:
        """
        self.sync_transact("SetJobMappingStorageContract", [job_id, storage_contract_address])

    def get_job_mapping_storage_contract(self, job_id: str) -> Address:
        return self.call("GetJobMappingStorageContract", [job_id])

    def set_job_status(self, job_id: str, status: int) -> Any:
        """

        :param job_id:
        :param status: 0->prepared, 1->running, 2->finished
        :return:
        """
        self.sync_transact("SetJobStatus", [job_id, status])

    def get_job_status(self, job_id: str) -> int:
        self.call("GetJobStatus", [job_id])



class StorageContract(Contract):

    def __init__(self):
        super(StorageContract, self).__init__()

    def get_fed_step(self) -> int:
        return self.call("GetFedStep")

    def update_activate_clients(self, client_address: Address, status: int) -> Any:
        """

        :param client_id:
        :param status: 0->join, 1->exit
        :return:
        """
        self.sync_transact("UpdateActivateClients", [client_address, status])

    def upload_model_parameters(self, fed_step: int, client_address: Address, model_pars_ipfs_hash: str) -> Any:
        return self.sync_transact("UploadModelParameters", [fed_step, client_address, model_pars_ipfs_hash])

    def selection_client_address(self, fed_step: int) -> Any:
        return self.sync_transact("SelectionClientAddress", [fed_step])

    def get_selection_client_address(self, fed_step) -> Address:
        return self.call("GetSelectionClientAddress", [fed_step])

    def download_model_parameters_ipfs_hash(self, fed_step: int, client_address: Address) -> Any:
        return self.call("DownloadModelParametersIpfsHash", [fed_step, client_address])

    def upload_aggregated_parameters_ipfs_hash(self, fed_step: int, client_address: Address, aggregated_model_pars_ipfs_hash: str) -> Any:
        return self.sync_transact("UploadAggreagtedParametersIpfsHash", [fed_step, client_address, aggregated_model_pars_ipfs_hash])

    def download_aggregated_parameters_ipfs_hash(self, fed_step: int) -> str:
        return self.call("DownloadAggregatedParametersIpfsHash", [fed_step])

    def download_final_aggregated_parameters_ipfs_hash(self) -> str:
        return self.call("DownloadFinalAggregatedParametersIpfsHash")

    def upload_final_aggregated_parameters_ipfs_hash(self, final_aggregated_ipfs_hash: str) -> Any:
        return self.sync_transact("UploadFinalAggregatedParametersIpfsHash" [final_aggregated_ipfs_hash])







