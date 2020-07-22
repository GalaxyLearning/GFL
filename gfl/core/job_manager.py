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

import torch
import threading
import pickle
import logging
import os, json
import inspect
from gfl.entity import runtime_config
from gfl.entity.job import Job
from gfl.core.exception import GFLException
from gfl.utils.utils import JobUtils, LoggerFactory
from gfl.core.strategy import WorkModeStrategy, FederateStrategy
from gfl.path import PathFactory

lock = threading.RLock()



class JobManager(object):
    """
    JobManager provides job related operations
    """

    def __init__(self):
        self.job_path = PathFactory.get_job_server_dir_path()
        self.logger = LoggerFactory.getLogger("JobManager", logging.INFO)

    def generate_job(self, fed_strategy=FederateStrategy.FED_AVG, epoch=0, model=None, distillation_alpha=None, l2_dist=False):
        """
        Generate job with user-defined strategy
        :param work_mode:
        :param train_strategy:
        :param fed_strategy:
        :param model:
        :param distillation_alpha:
        :return: job object
        """
        with lock:
            # server_host, job_id, train_strategy, train_model, train_model_class_name, fed_strategy, iterations, distillation_alpha
            if fed_strategy == FederateStrategy.FED_DISTILLATION and distillation_alpha is None:
                raise GFLException("generate_job() missing 1 positoonal argument: 'distillation_alpha'")
            if epoch == 0:
                raise GFLException("generate_job() missing 1 positoonal argument: 'epoch'")

            job = Job(server_host=None, job_id=JobUtils.generate_job_id(), train_model=inspect.getsourcefile(model),
                      train_model_class_name=model.__name__, aggregate_strategy=fed_strategy, epoch=epoch,  distillation_alpha=distillation_alpha, l2_dist=l2_dist)

            # if work_mode == WorkModeStrategy.WORKMODE_STANDALONE:
            #     job.set_server_host("localhost:8080")
            # else:
            #     job.set_server_host("")

            return job

    def submit_job(self, job, model):
        """
        Submit job
        :param job:
        :param model:
        :return:
        """
        with lock:
            # create model dir of this job
            torch.save(model.state_dict(), PathFactory.get_init_model_pars_path(job.get_job_id()))

            init_model_path = PathFactory.get_job_init_model_code_path(job.get_job_id())
            with open(init_model_path, "w") as model_f:
                with open(job.get_train_model(), "r") as model_f2:
                    for line in model_f2.readlines():
                        model_f.write(line)
            if not os.path.exists(self.job_path):
                os.mkdir(self.job_path)
            with open(os.path.join(self.job_path, "job_{}".format(job.get_job_id())), "wb") as f:
                pickle.dump(job, f)

            self.logger.info("job {} added successfully".format(job.get_job_id()))

    def prepare_job(self, job):
        with lock:
            runtime_config.remove_waiting_job(job)
            runtime_config.add_pending_job(job)

    def exec_job(self, job):
        with lock:
            exec_job = runtime_config.remove_pending_job(job)
            runtime_config.add_exec_job(job)

    def complete(self):
        with lock:
            runtime_config.get_exec_job()

    @staticmethod
    def get_job_list(job_path):
        job_list = []
        for job_file in os.listdir(job_path):
            job_file_path = os.path.join(job_path, job_file)
            with open(job_file_path, "rb") as f:
                job = pickle.load(f)
                job_list.append(job)
        return job_list
