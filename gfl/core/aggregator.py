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

LOCAL_AGGREGATE_FILE = os.path.join("tmp_aggregate_pars", "avg_pars")


class Aggregator(object):
    def __init__(self):
        pass

    def aggregate(self, job_model_pars, base_model_path, job_id, fed_step):
        avg_model_par = self._aggregate_exec(job_model_pars, base_model_path, job_id, fed_step)
        tmp_aggregate_dir = os.path.join(base_model_path, "models_{}".format(job_id))
        tmp_aggregate_path = os.path.join(base_model_path, "models_{}".format(job_id),
                                          "{}_{}".format(LOCAL_AGGREGATE_FILE, fed_step))
        if not os.path.exists(tmp_aggregate_dir):
            os.makedirs(tmp_aggregate_dir)
        torch.save(avg_model_par, tmp_aggregate_path)


    def _aggregate_exec(self, job_model_pars, base_model_path, job_id, fed_step):
        pass


class FedAvgAggregator(Aggregator):
    """
    FedAvgAggregator is responsible for aggregating model parameters by using FedAvg Algorithm
    """

    def __init__(self):
        super(FedAvgAggregator, self).__init__()
        self.logger = LoggerFactory.getLogger("FedAvgAggregator", logging.INFO)

    def _aggregate_exec(self, job_model_pars, base_model_path, job_id, fed_step):
        avg_model_par = job_model_pars[0]
        for key in avg_model_par.keys():
            for i in range(1, len(job_model_pars)):
                avg_model_par[key] += job_model_pars[i][key]
            avg_model_par[key] = torch.div(avg_model_par[key], len(job_model_pars))

        self.logger.info("job: {} the {}th round parameters aggregated successfully!".format(job_id, fed_step))
        return avg_model_par