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
from gfl.utils.utils import LoggerFactory
from gfl.path import PathFactory

LOCAL_AGGREGATE_FILE = os.path.join("tmp_aggregate_pars", "avg_pars")


class Aggregator(object):
    def __init__(self):
        pass

    def aggregate(self, job_model_pars, base_model_path, job_id, fed_step):
        avg_model_par = self._aggregate_exec(job_model_pars, base_model_path, job_id, fed_step)
        tmp_aggregate_path = os.path.join(PathFactory.get_job_model_path(job_id),
                                          "{}_{}".format(LOCAL_AGGREGATE_FILE, fed_step))
        torch.save(avg_model_par, tmp_aggregate_path)


    def _aggregate_exec(self, job_model_pars, base_model_path, job_id, fed_step):
        pass

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