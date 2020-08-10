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
from gfl.exceptions.fl_expection import GFLException
from gfl.utils.utils import JobUtils, LoggerFactory, ModelUtils
from gfl.core.strategy import WorkModeStrategy, FederateStrategy
from gfl.core.trainer import TrainStandloneNormalStrategy, TrainMPCNormalStrategy, \
    TrainStandloneDistillationStrategy, TrainMPCDistillationStrategy

JOB_PATH = os.path.join(os.path.abspath("."), "res", "jobs_client")


class TrainerController(object):
    """
    TrainerController is responsible for choosing a apprpriate train strategy for corresponding job
    """

    def __init__(self, work_mode=WorkModeStrategy.WORKMODE_STANDALONE, models=None, data=None, client_id=0,
                 client_ip="",
                 client_port=8081, server_url="", curve=False, local_epoch=5, concurrent_num=5):
        self.work_mode = work_mode
        self.data = data
        self.client_id = str(client_id)
        self.local_epoch = local_epoch
        self.concurrent_num = concurrent_num
        self.trainer_executor_pool = ThreadPoolExecutor(self.concurrent_num)
        self.job_path = JOB_PATH
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
            self._trainer_standalone_exec()
        else:
            response = requests.post(
                "/".join([self.server_url, "register", self.client_ip, '%s' % self.client_port, '%s' % self.client_id]))
            response_json = response.json()
            if response_json['code'] == 200 or response_json['code'] == 201:
                self.trainer_executor_pool.submit(communicate_client.start_communicate_client, self.client_ip,
                                                  self.client_port)
                self._trainer_mpc_exec()
            else:
                GFLException("connect to parameter server fail, please check your internet")

    def _trainer_standalone_exec(self):
        t = threading.Timer(5, self._trainer_standalone_exec_impl)
        t.start()

    def _trainer_standalone_exec_impl(self):
        self.logger.info("searching for new jobs...")
        JobUtils.get_job_from_remote(None, self.job_path)
        job_list = JobUtils.list_all_jobs(self.job_path)
        for job in job_list:
            if self.job_train_strategy.get(job.get_job_id()) is None:
                # print(job.get_aggregate_strategy())
                gfl_model = ModelUtils.get_model_by_job_id(self.models, job.get_job_id())
                if job.get_aggregate_strategy() == FederateStrategy.FED_AVG.value:
                    self.job_train_strategy[job.get_job_id()] = TrainStandloneNormalStrategy(job, self.data,
                                                                                             self.fed_step,
                                                                                             self.client_id,
                                                                                             self.local_epoch,
                                                                                             gfl_model,
                                                                                             self.curve)
                else:
                    self.job_train_strategy[job.get_job_id()] = TrainStandloneDistillationStrategy(job, self.data,
                                                                                                   self.fed_step,
                                                                                                   self.client_id,
                                                                                                   self.local_epoch,
                                                                                                   gfl_model,
                                                                                                   self.curve)
                self.run(self.job_train_strategy.get(job.get_job_id()))

    def _trainer_mpc_exec(self):
        t = threading.Timer(5, self._trainer_mpc_exec_impl)
        t.start()

    def _trainer_mpc_exec_impl(self):
        self.logger.info("searching for new jobs...")
        JobUtils.get_job_from_remote(self.server_url, self.job_path)
        job_list = JobUtils.list_all_jobs(self.job_path)
        for job in job_list:
            if self.job_train_strategy.get(job.get_job_id()) is None:
                gfl_model = ModelUtils.get_model_by_job_id(self.models, job.get_job_id())
                if job.get_aggregate_strategy() == FederateStrategy.FED_AVG.value:
                    self.job_train_strategy[job.get_job_id()] = self.job_train_strategy[
                        job.get_job_id()] = TrainMPCNormalStrategy(job, self.data, self.fed_step, self.client_ip,
                                                                   self.client_port, self.server_url, self.client_id, self.local_epoch,
                                                                   gfl_model, self.curve)
                else:
                    self.job_train_strategy[job.get_job_id()] = TrainMPCDistillationStrategy(job, self.data,
                                                                                             self.fed_step,
                                                                                             self.client_ip,
                                                                                             self.client_port,
                                                                                             self.server_url,
                                                                                             self.client_id, self.local_epoch,
                                                                                             gfl_model,
                                                                                             self.curve)
                self.run(self.job_train_strategy.get(job.get_job_id()))

    def run(self, trainer):
        trainer.train()

