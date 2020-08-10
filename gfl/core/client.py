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
import importlib
from gfl.utils.utils import JobUtils
from gfl.entity.model import Model

JOB_PATH = os.path.join(os.path.abspath("."), "res", "jobs_client")
BASE_MODEL_PATH = os.path.join(os.path.abspath("."), "res", "models")


class FLClient(object):
    def __init__(self):
        super(FLClient, self).__init__()
        self.job_path = JOB_PATH
        self.base_model_path = BASE_MODEL_PATH


    def get_remote_gfl_models(self, server_url=None):
        if server_url is None:
            return self._get_models_from_local()
        else:
            return self._get_models_from_remote(server_url)


    def _get_models_from_local(self):
        model_list = []
        JobUtils.get_job_from_remote(None, self.job_path)
        job_list = JobUtils.list_all_jobs(self.job_path)

        for job in job_list:
            model = self._get_model_from_job(job)
            gfl_model = Model()
            gfl_model.set_model(model)
            gfl_model.set_job_id(job.get_job_id())
            model_list.append(gfl_model)
        return model_list

    def _get_models_from_remote(self, server_url):
        model_list = []
        JobUtils.get_job_from_remote(server_url, self.job_path)
        job_list = JobUtils.list_all_jobs(self.job_path)
        for job in job_list:
            model = self._get_model_from_job(job)
            gfl_model = Model()
            gfl_model.set_model(model)
            gfl_model.set_job_id(job.get_job_id())
            model_list.append(gfl_model)
        return model_list

    def _get_model_from_job(self, job):
        job_id = job.get_job_id()
        module = importlib.import_module("res.models.models_{}.init_model_{}".format(job_id, job_id),
                                         "init_model_{}".format(job_id))
        model_class = getattr(module, job.get_train_model_class_name())
        return model_class()




