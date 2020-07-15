# Copyright (c) 2020 GalaxyLearning Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from web3 import Web3
from pfl.utils.utils import LoggerFactory
from pfl.exceptions.fl_expection import PFLException
from pfl.contract.pfl_storage_contract import PFLStorage
from pfl.contract.pfl_controller_contract import PFLController
from pfl.contract.pfl_storage_contract import PFLStorage

class PFLEthereumUtils:

    logger = LoggerFactory.getLogger("PFLEthereumUtils", logging.INFO)
    pfl_controller_contract = None

    @staticmethod
    def get_connection_with_ethereum(url=None):

        if url is None:
            raise PFLException("get_connection_with_ethereum() missing 1 positional argument")

        try:
            web3 = Web3(Web3.HTTPProvider(url))
        except Exception:
            raise PFLException("Connect to ethereum fail!!!")
        else:
            return web3

    @staticmethod
    def init_pfl_controller_contracts(web3=None, account=None, account_password=None, pfl_controller_contract_address=None):

        if PFLEthereumUtils.pfl_controller_contract:
            return

        if web3 is None or account is None or account_password is None:
            raise PFLException("init_ethereum_contracts() missing positional arguments")

        if not PFLEthereumUtils.pfl_controller_contract:
            PFLEthereumUtils.pfl_controller_contract = PFLController(web3=web3, account=account, password=account_password, contract_address=pfl_controller_contract_address)



    @staticmethod
    def get_pfl_controller_contract_instance():
        if PFLEthereumUtils.pfl_controller_contract is None:
            raise PFLException("Ethereum contracts is not initialization, please inoke init_ethereum_contracts(web3) first!")
        return PFLEthereumUtils.pfl_controller_contract

    @staticmethod
    def deploy_and_get_pfl_storage_contract_instance(web3=None, account=None, password=None):
        if web3 is None or account is None or password is None:
            raise PFLException("init_ethereum_contracts() missing positional arguments")
        pfl_storage = PFLStorage(web3=web3, account=account, password=password)
        return pfl_storage

    @staticmethod
    def get_pfl_storage_contract_instance(web3=None, pfl_storage_contract_address=None):
        if web3 is None or pfl_storage_contract_address is None:
            raise PFLException("init_ethereum_contracts() missing 1 positional argument")

        pfl_storage = PFLStorage(web3, pfl_storage_contract_address)
        return pfl_storage

    @staticmethod
    def block_support_function():
        PFLEthereumUtils.logger.info("ethereum connection still alive")