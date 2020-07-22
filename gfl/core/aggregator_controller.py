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
import torch
import time
import requests
import logging
from concurrent.futures import ThreadPoolExecutor
from gfl.core.strategy import WorkModeStrategy, FederateStrategy
from gfl.core.job_manager import JobManager
from gfl.utils.utils import LoggerFactory, RuntimeConfigUtils
from gfl.entity.runtime_config import RuntimeServerConfig
from gfl.core.aggregator import FedAvgAggregator
from gfl.path import PathFactory


class AggregatorController(object):
    """
    Aggregator is responsible for aggregating model parameters

    """
    def __init__(self, work_mode, job_path, base_model_path, concurrent_num=5):
        self.job_path = job_path
        self.base_model_path = base_model_path
        self.aggregate_executor_pool = ThreadPoolExecutor(concurrent_num)
        self.work_mode = work_mode
        self.fed_step = {}
        self.fed_avg_aggregator = FedAvgAggregator()
        self.logger = LoggerFactory.getLogger("Aggregator", logging.INFO)

    def load_model_pars(self, job_model_pars_path, fed_step):
        """

        :param job_model_pars_path:
        :param fed_step:
        :return:
        """
        fed_step = 0 if fed_step is None else fed_step
        job_model_pars = []
        last_model_par_file_num = 0
        # print("job_model_pars_path: ", job_model_pars_path)
        for f in os.listdir(job_model_pars_path):
            if f.find("models_") != -1:
                one_model_par_path = os.path.join(job_model_pars_path, f)
                # print("one_model_par_path: ", one_model_par_path)
                one_model_par_files = os.listdir(one_model_par_path)
                if one_model_par_files and len(one_model_par_files) != 0:
                    last_model_par_file_num = self._find_last_model_file_num(one_model_par_files)
                    if last_model_par_file_num > fed_step:
                        model_par = torch.load(os.path.join(one_model_par_path, one_model_par_files[-1]))
                        job_model_pars.append(model_par)
                    else:
                        return None, 0
                else:
                    # wait for other clients finish training
                    return None, 0

        return job_model_pars, last_model_par_file_num

    def _find_last_model_file_num(self, files):
        last_num = 0
        for file in files:
            file_num = int(file.split("_")[-1])
            last_num = file_num if last_num < file_num else last_num
        return last_num

    def _broadcast(self, job_id, connected_client_list, base_model_path):
        """

        :param job_id_list:
        :param connected_client_list:
        :param base_model_path:
        :return:
        """
        aggregated_files = self._prepare_upload_aggregate_file(job_id, base_model_path)
        self.logger.info("connected client list: {}".format(connected_client_list))
        for client in connected_client_list:
            client_url = "http://{}".format(client)
            response = requests.post("/".join([client_url, "aggregatepars"]), data=None, files=aggregated_files)
            # print(response)

    def _prepare_upload_aggregate_file(self, job_id, base_model_path):
        """

        :param job_id_list:
        :param base_model_path:
        :return:
        """
        aggregated_files = {}

        tmp_aggregate_dir = os.path.join(base_model_path, "models_{}".format(job_id), "tmp_aggregate_pars")
        fed_step = self._find_last_model_file_num(os.listdir(tmp_aggregate_dir))
        send_aggregate_filename = "tmp_aggregate_{}_{}".format(job_id, fed_step)
        tmp_aggregate_path = os.path.join(tmp_aggregate_dir, "avg_pars_{}".format(fed_step))
        aggregated_files[send_aggregate_filename] = (send_aggregate_filename, open(tmp_aggregate_path, "rb"))
        return aggregated_files

    def _save_final_model_pars(self, job_id, tmp_aggregate_dir, fed_step):
        """

        :param job_id:
        :param tmp_aggregate_dir:
        :param fed_step:
        :return:
        """
        job_model_dir = os.path.join(self.base_model_path, "models_{}".format(job_id))
        final_model_pars_path = os.path.join(os.path.abspath("."), "final_model_pars_{}".format(job_id))
        last_aggregate_file = os.path.join(tmp_aggregate_dir, "avg_pars_{}".format(fed_step))
        with open(final_model_pars_path, "wb") as final_f:
            with open(last_aggregate_file, "rb") as f:
                for line in f.readlines():
                    final_f.write(line)

        self.logger.info("job {} save final aggregated parameters successfully!".format(job_id))


    def start(self):
        pass

    def aggregate(self, job):
        pass

    def _exec(self, aggregator, job_model_pars, base_model_path, job_id, fed_step):
        return aggregator.aggregate(job_model_pars, base_model_path, job_id, fed_step)


class StandaloneAggregatorController(AggregatorController):
    def __init__(self, job_path, base_model_path, concurrent_num=5):
        super(StandaloneAggregatorController, self).__init__(job_path, base_model_path, concurrent_num)


    def start(self):
        job_list = JobManager.get_job_list(self.job_path)
        for job in job_list:
            self.aggregate(job)

    def aggregate(self, job):

        job_model_pars, fed_step = self.load_model_pars(
            os.path.join(self.base_model_path, "models_{}".format(job.get_job_id())),
            self.fed_step.get(job.get_job_id()))
        # print("fed_step: {}, self.fed_step: {}, job_model_pars: {}".format(fed_step, self.fed_step.get(job.get_job_id()), job_model_pars))
        job_fed_step = 0 if self.fed_step.get(job.get_job_id()) is None else self.fed_step.get(job.get_job_id())
        if job_fed_step != fed_step and job_model_pars is not None:

            self.logger.info("Aggregating......")
            if job.get_aggregate_strategy() == FederateStrategy.FED_AVG:
                aggregated_model_pars = self._exec(self.fed_avg_aggregator, job_model_pars, self.base_model_path, job.get_job_id(),
                           fed_step)
            self.fed_step[job.get_job_id()] = fed_step

            if job.get_epoch() <= self.fed_step[job.get_job_id()]:
                self._save_final_model_pars(job.get_job_id(), os.path.join(self.base_model_path,
                                                                           "models_{}".format(job.get_job_id()),
                                                                           "tmp_aggregate_pars"),
                                            self.fed_step[job.get_job_id()])
            return aggregated_model_pars


class ClusterAggregatorController(AggregatorController):
    def __init__(self, job_path, base_model_path, concurrent_num=5):
        super(ClusterAggregatorController, self).__init__(job_path, base_model_path, concurrent_num)


    def start(self):
        job_list = JobManager.get_job_list(self.job_path)
        # WAITING_BROADCAST_AGGREGATED_JOB_ID_LIST.clear()
        for job in job_list:
            self.aggregate(job)

    def aggregate(self, job):
        # job_list = JobManager.get_job_list(self.job_path)
        # # WAITING_BROADCAST_AGGREGATED_JOB_ID_LIST.clear()
        # for job in job_list:
        job_model_pars, fed_step = self.load_model_pars(
            os.path.join(self.base_model_path, "models_{}".format(job.get_job_id())),
            self.fed_step.get(job.get_job_id()))
        # print("fed_step: {}, self.fed_step: {}, job_model_pars: {}".format(fed_step, self.fed_step.get(job.get_job_id()), job_model_pars))
        job_fed_step = 0 if self.fed_step.get(job.get_job_id()) is None else self.fed_step.get(job.get_job_id())
        if job_fed_step != fed_step and job_model_pars is not None:

            self.logger.info("Aggregating......")
            if job.get_aggregate_strategy() == FederateStrategy.FED_AVG:
                aggregated_model_pars = self._exec(self.fed_avg_aggregator, job_model_pars, self.base_model_path, job.get_job_id(),
                           fed_step)
            self.fed_step[job.get_job_id()] = fed_step
            # WAITING_BROADCAST_AGGREGATED_JOB_ID_LIST.append(job.get_job_id())
            if job.get_epoch() <= self.fed_step[job.get_job_id()]:
                self._save_final_model_pars(job.get_job_id(), os.path.join(self.base_model_path,
                                                                           "models_{}".format(job.get_job_id()),
                                                                           "tmp_aggregate_pars"),
                                            self.fed_step[job.get_job_id()])

            runtime_server_config = RuntimeConfigUtils.get_obj_from_runtime_config_file(PathFactory.get_runtime_config_server_path(),
                                                                                        RuntimeServerConfig)
            self._broadcast(job.get_job_id(), runtime_server_config.CONNECTED_TRAINER_LIST, self.base_model_path)

            return aggregated_model_pars



