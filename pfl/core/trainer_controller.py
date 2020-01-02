import os
import threading
import requests
import logging
from concurrent.futures import ThreadPoolExecutor
from pfl.core import communicate_client
from pfl.utils.utils import JobUtils, LoggerFactory
from pfl.core.strategy import WorkModeStrategy, FederateStrategy
from pfl.core.trainer import TrainStandloneNormalStrategy, TrainMPCNormalStrategy, \
    TrainStandloneDistillationStrategy, TrainMPCDistillationStrategy

JOB_PATH = os.path.join(os.path.abspath("."), "res", "jobs_client")


class TrainerController(object):
    def __init__(self, work_mode, data, client_id, client_ip, client_port, server_url, concurrent_num=5):
        self.work_mode = work_mode
        self.data = data
        self.client_id = client_id
        self.concurrent_num = concurrent_num
        self.trainer_executor_pool = ThreadPoolExecutor(self.concurrent_num)
        self.job_path = JOB_PATH
        self.fed_step = {}
        self.job_iter_dict = {}
        self.job_train_strategy = {}
        self.is_finish = True
        self.client_ip = client_ip
        self.client_port = client_port
        self.server_url = server_url
        self.logger = LoggerFactory.getLogger("TrainerController", logging.INFO)

    def start(self):
        if self.work_mode == WorkModeStrategy.WORKMODE_STANDALONE:
            self._trainer_standalone_exec()
            # self._trainer_standalone_exec()
        else:
            response = requests.post(
                "/".join([self.server_url, "register", self.client_ip, '%s' % self.client_port, '%s' % self.client_id]))
            response_json = response.json()
            if response_json['code'] == 200 or response_json['code'] == 201:
                # self.trainer_executor_pool.submit(communicate_client.start_communicate_client, self.client_ip, self.client_port)
                # self.trainer_executor_pool.submit(self._trainer_mpc_exec, self.server_url)
                # communicate_client.start_communicate_client(self.client_ip, int(self.client_port))
                self.trainer_executor_pool.submit(communicate_client.start_communicate_client, self.client_ip,
                                                  self.client_port)
                self._trainer_mpc_exec()
            else:
                print("connect to parameter server fail, please check your internet")

    def _trainer_standalone_exec(self):
        t = threading.Timer(5, self._trainer_standalone_exec_impl)
        t.start()

    def _trainer_standalone_exec_impl(self):
        JobUtils.get_job_from_remote(None, self.job_path)
        job_list = JobUtils.list_all_jobs(self.job_path, self.job_iter_dict)
        for job in job_list:
            if self.job_train_strategy.get(job.get_job_id()) is None:
                #print(job.get_aggregate_strategy())
                if job.get_aggregate_strategy() == FederateStrategy.FED_AVG.value:
                    self.job_train_strategy[job.get_job_id()] = TrainStandloneNormalStrategy(job, self.data,
                                                                                             self.fed_step,
                                                                                             self.client_id)
                else:
                    self.job_train_strategy[job.get_job_id()] = TrainStandloneDistillationStrategy(job, self.data,
                                                                                                   self.fed_step,
                                                                                                   self.client_id)
                self.run(self.job_train_strategy.get(job.get_job_id()))

    def _trainer_mpc_exec(self):
        t = threading.Timer(5, self._trainer_mpc_exec_impl)
        t.start()

    def _trainer_mpc_exec_impl(self):

        JobUtils.get_job_from_remote(self.server_url, self.job_path)
        job_list = JobUtils.list_all_jobs(self.job_path, self.job_iter_dict)
        for job in job_list:
            if self.job_train_strategy.get(job.get_job_id()) is None:
                if job.get_aggregate_strategy() == FederateStrategy.FED_AVG.value:
                    self.job_train_strategy[job.get_job_id()] = self.job_train_strategy[
                        job.get_job_id()] = TrainMPCNormalStrategy(job, self.data, self.fed_step, self.client_ip,
                                                                   self.client_port, self.server_url, self.client_id)
                else:
                    self.job_train_strategy[job.get_job_id()] = TrainMPCDistillationStrategy(job, self.data,
                                                                                             self.fed_step,
                                                                                             self.client_ip,
                                                                                             self.client_port,
                                                                                             self.server_url,
                                                                                             self.client_id)
                self.run(self.job_train_strategy.get(job.get_job_id()))

    def run(self, trainer):
        trainer.train()
