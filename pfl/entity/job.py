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


class Job(object):

    def __init__(self, server_host, job_id, train_model, train_model_class_name, aggregate_strategy, epoch,
                 distillation_alpha=None, l2_dist=False):
        self.server_host = server_host
        self.job_id = job_id
        self.epoch = epoch
        self.train_model = train_model
        self.train_model_class_name = train_model_class_name
        self.aggregate_strategy = aggregate_strategy
        self.alpha = distillation_alpha
        self.l2_dist = l2_dist

    def set_server_host(self, server_host):
        self.server_host = server_host

    def set_job_id(self, job_id):
        self.job_id = job_id

    def get_job_id(self):
        return self.job_id

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_train_model(self, train_model):
        self.train_model = train_model

    def set_train_model_class_name(self, train_model_class_name):
        self.train_model_class_name = train_model_class_name

    def get_train_model_class_name(self):
        return self.train_model_class_name

    def get_server_host(self):
        return self.server_host

    def get_epoch(self):
        return self.epoch

    def get_train_model(self):
        return self.train_model

    def set_aggregate_stragety(self, aggregate_strategy):
        self.aggregate_strategy = aggregate_strategy

    def get_aggregate_strategy(self):
        return self.aggregate_strategy

    def set_distillation_alpha(self, alpha):
        self.alpha = alpha

    def get_distillation_alpha(self):
        return self.alpha

    def set_l2_dist(self, l2_dist):
        self.l2_dist = l2_dist

    def get_l2_dist(self):
        return self.l2_dist
