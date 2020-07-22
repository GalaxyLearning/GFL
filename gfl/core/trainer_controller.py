# Copyright (c) 2019 GalaxyLearning Authors. All Rights Reserved.
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

import os
import requests
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from gfl.core import communicate_client
from gfl.core.exception import GFLException
from gfl.utils.json_utils import JsonUtil
from gfl.entity.runtime_config import RuntimeClientConfig
from gfl.utils.utils import JobUtils, LoggerFactory, ModelUtils
from gfl.core.strategy import WorkModeStrategy, FederateStrategy
from gfl.path import PathFactory
from gfl.core.trainer import TrainStandloneNormalStrategy, TrainMPCNormalStrategy, \
    TrainStandloneDistillationStrategy, TrainMPCDistillationStrategy, TrainBlockchainNormalSelectionStrategy, TrainBlockchainDistillationSelectionStrategy



class TrainerControllerBase(object):
    def __init__(self):
        runtime_server_config_json = JsonUtil.to_json(RuntimeClientConfig())

        with open(PathFactory.get_runtime_config_client_path(), "w") as f:
            f.write(runtime_server_config_json)


class TrainerController(TrainerControllerBase):
    """
    TrainerController is responsible for choosing a apprpriate train strategy for corresponding job
    """

    def __init__(self, work_mode=WorkModeStrategy.WORKMODE_STANDALONE, models=None, data=None, client_id=0,
                 client_ip="",
                 client_port=8081, server_url="", curve=False, local_epoch=5, concurrent_num=5):
        super(TrainerController, self).__init__()
        self.work_mode = work_mode
        self.data = data
        self.client_id = str(client_id)
        self.local_epoch = local_epoch
        self.concurrent_num = concurrent_num
        self.trainer_executor_pool = ThreadPoolExecutor(self.concurrent_num)
        self.job_path = PathFactory.get_job_client_dir_path()
        self.models = models
        self.fed_step = {}
        self.job_train_strategy = {}
        self.client_ip = client_ip
        self.client_port = str(client_port)
        self.server_url = server_url
        self.curve = curve
        self.logger = LoggerFactory.getLogger("TrainerController", logging.INFO)



    def start(self):
        if self.work_mode == WorkModeStrategy.WORKMODE_STANDALONE:
            self.__trainer_standalone_exec()
        elif self.work_mode == WorkModeStrategy.WORKMODE_CLUSTER:
            response = requests.post(
                "/".join([self.server_url, "register", self.client_ip, '%s' % self.client_port, '%s' % self.client_id]))
            response_json = response.json()
            if response_json['code'] == 200 or response_json['code'] == 201:
                self.trainer_executor_pool.submit(communicate_client.start_communicate_client, self.client_ip,
                                                  self.client_port)
                self.__trainer_mpc_exec()
            else:
                GFLException("connect to parameter server fail, please check your internet")
        else:
            self.__trainer_blockchain_exec()

    def __trainer_standalone_exec(self):
         self.trainer_executor_pool.submit(self.__trainer_standalone_exec_impl)


    def __trainer_standalone_exec_impl(self):
        self.logger.info("searching for new jobs...")
        JobUtils.get_job_from_remote(None, self.job_path)
        job_list = JobUtils.list_all_jobs(self.job_path)
        for job in job_list:
            if self.job_train_strategy.get(job.get_job_id()) is None:
                # print(job.get_aggregate_strategy())
                pfl_model = ModelUtils.get_model_by_job_id(self.models, job.get_job_id())
                if job.get_aggregate_strategy() == FederateStrategy.FED_AVG.value:
                    self.job_train_strategy[job.get_job_id()] = TrainStandloneNormalStrategy(job, self.data,
                                                                                             self.fed_step,
                                                                                             self.client_id,
                                                                                             self.local_epoch,
                                                                                             pfl_model,
                                                                                             self.curve)
                else:
                    self.job_train_strategy[job.get_job_id()] = TrainStandloneDistillationStrategy(job, self.data,
                                                                                                   self.fed_step,
                                                                                                   self.client_id,
                                                                                                   self.local_epoch,
                                                                                                   pfl_model,
                                                                                                   self.curve)
                self.run(self.job_train_strategy.get(job.get_job_id()))

    def __trainer_mpc_exec(self):
        self.trainer_executor_pool.submit(self._trainer_mpc_exec_impl)


    def __trainer_mpc_exec_impl(self):
        self.logger.info("searching for new jobs...")
        JobUtils.get_job_from_remote(self.server_url, self.job_path)
        job_list = JobUtils.list_all_jobs(self.job_path)
        for job in job_list:
            if self.job_train_strategy.get(job.get_job_id()) is None:
                pfl_model = ModelUtils.get_model_by_job_id(self.models, job.get_job_id())
                if job.get_aggregate_strategy() == FederateStrategy.FED_AVG.value:
                    self.job_train_strategy[job.get_job_id()] = self.job_train_strategy[
                        job.get_job_id()] = TrainMPCNormalStrategy(job, self.data, self.fed_step, self.client_ip,
                                                                   self.client_port, self.server_url, self.client_id, self.local_epoch,
                                                                   pfl_model, self.curve)
                else:
                    self.job_train_strategy[job.get_job_id()] = TrainMPCDistillationStrategy(job, self.data,
                                                                                             self.fed_step,
                                                                                             self.client_ip,
                                                                                             self.client_port,
                                                                                             self.server_url,
                                                                                             self.client_id, self.local_epoch,
                                                                                             pfl_model,
                                                                                             self.curve)
                self.run(self.job_train_strategy.get(job.get_job_id()))


    def __trainer_blockchain_exec(self):
        self.trainer_executor_pool.submit(self.__trainer_blockchain_exec_impl)

    def __trainer_blockchain_exec_impl(self):
        self.logger.info("searching for new jobs...")
        JobUtils.get_job_from_remote(self.server_url, self.job_path)
        job_list = JobUtils.list_all_jobs(self.job_path)
        for job in job_list:
            if not JobUtils.is_job_completed(job.get_job_id()):
                pfl_model = ModelUtils.get_model_by_job_id(self.models, job.get_job_id())
                if job.get_aggregate_strategy() == FederateStrategy.FED_AVG.value:
                    self.job_train_strategy[job.get_job_id()] = self.job_train_strategy[
                        job.get_job_id()] = TrainBlockchainNormalSelectionStrategy(job, self.data, self.fed_step, self.client_ip,
                                                                   self.client_port, self.server_url, self.client_id,
                                                                   self.local_epoch,
                                                                   pfl_model, self.curve)
                else:
                    self.job_train_strategy[job.get_job_id()] = TrainBlockchainDistillationSelectionStrategy(job, self.data,
                                                                                             self.fed_step,
                                                                                             self.client_ip,
                                                                                             self.client_port,
                                                                                             self.server_url,
                                                                                             self.client_id,
                                                                                             self.local_epoch,
                                                                                             pfl_model,
                                                                                             self.curve)
                self.run(self.job_train_strategy.get(job.get_job_id()))

    def run(self, trainer):
        trainer.train()




