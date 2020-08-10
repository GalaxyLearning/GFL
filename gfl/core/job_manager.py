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
from gfl.exceptions.fl_expection import GFLException
from gfl.utils.utils import JobUtils, LoggerFactory
from gfl.core.strategy import WorkModeStrategy, FederateStrategy

lock = threading.RLock()

JOB_PATH = os.path.join(os.path.abspath("."), "res", "jobs_server")
MODEL_PATH = os.path.join(os.path.abspath("."), "res", "models")


class JobManager(object):
    """
    JobManager provides job related operations
    """

    def __init__(self):
        self.job_path = JOB_PATH
        self.logger = LoggerFactory.getLogger("JobManager", logging.INFO)

    def generate_job(self, work_mode=WorkModeStrategy.WORKMODE_STANDALONE,
                     fed_strategy=FederateStrategy.FED_AVG, epoch=0, model=None, distillation_alpha=None, l2_dist=False):
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

            job = Job(None, JobUtils.generate_job_id(), inspect.getsourcefile(model),
                      model.__name__, fed_strategy, epoch,  distillation_alpha=distillation_alpha, l2_dist=l2_dist)

            if work_mode == WorkModeStrategy.WORKMODE_STANDALONE:
                job.set_server_host("localhost:8080")
            else:
                job.set_server_host("")

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
            job_model_dir = os.path.join(MODEL_PATH, "models_{}".format(job.get_job_id()))
            if not os.path.exists(job_model_dir):
                os.makedirs(job_model_dir)
            torch.save(model.state_dict(), os.path.join(job_model_dir, "init_model_pars_{}".format(job.get_job_id())))

            init_model_path = os.path.join(job_model_dir, "init_model_{}.py".format(job.get_job_id()))
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
