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
import json
import threading
import datetime
import pickle
import requests
import logging
from json.decoder import WHITESPACE

from gfl.entity.job import Job
from gfl.core.strategy import TrainStrategy

LOG_FILE = os.path.join(os.path.abspath("."), "log.txt")


class JobIdCount(object):
    """
    JobIdCount is responsible for generating a number synchrnized.
    """
    lock = threading.RLock()

    def __init__(self, init_value):
        self.value = init_value

    def incr(self, step):
        with JobIdCount.lock:
            self.value += step
            return self.value



jobCount = JobIdCount(init_value=0)


class JobUtils(object):
    """
    JobUtils provides some tool methods of operating jobs
    """
    def __init__(self):
        pass

    @staticmethod
    def generate_job_id():
        return '{}{}'.format(datetime.datetime.now().strftime("%Y%m%d%H%M%S%f"), jobCount.incr(1))

    @staticmethod
    def list_all_jobs(job_path):
        job_list = []
        for file in os.listdir(job_path):
            # print("job file: ", job_path+"\\"+file)
            with open(os.path.join(job_path, file), "rb") as f:
                job = pickle.load(f)
                job_list.append(job)
        return job_list

    @staticmethod
    def serialize(job):
        return pickle.dumps(job)

    @staticmethod
    def get_job_from_remote(server_url, job_path):
        """
        Get jobs from remote
        If server_url is None, searching jobs from local
        :param server_url:
        :param job_path:
        :return:
        """
        if not os.path.exists(job_path):
            os.mkdir(job_path)
        if server_url is None:
            job_server_path = os.path.join(os.path.dirname(job_path), "jobs_server")
            for file in os.listdir(job_server_path):
                with open(os.path.join(job_server_path, file), "rb") as job_f:
                    job = pickle.load(job_f)
                    job_str = json.dumps(job, cls=JobEncoder)
                    new_job = json.loads(job_str, cls=JobDecoder)
                    with open(os.path.join(job_path, file), "wb") as new_job_f:
                        pickle.dump(new_job, new_job_f)
        else:
            response = requests.get("/".join([server_url, "jobs"]))
            response_data = response.json()
            job_list_str = response_data['data']
            for job_str in job_list_str:
                job = json.loads(job_str, cls=JobDecoder)
                job_filename = os.path.join(job_path, "job_{}".format(job.get_job_id()))
                with open(job_filename, "wb") as job_f:
                    pickle.dump(job, job_f)


class ModelUtils(object):
    @staticmethod
    def get_model_by_job_id(models, job_id):
        for model in models:
            if model.get_job_id() == job_id:
                return model
        return None


class JobEncoder(json.JSONEncoder):

    def default(self, o):
        if isinstance(o, Job):
            return {
                'job_id': o.get_job_id(),
                'train_model': o.get_train_model(),
                'epoch': o.get_epoch(),
                'train_model_class_name': o.get_train_model_class_name(),
                'server_host': o.get_server_host(),
                'aggregate_strategy': o.get_aggregate_strategy().value,
                'distillation_alpha': o.get_distillation_alpha(),
                'l2_dist': o.get_l2_dist()
            }
        return json.JSONEncoder.default(self, o)


class JobDecoder(json.JSONDecoder):
    def decode(self, s, _w=WHITESPACE.match):
        dict = super().decode(s)
        return Job(dict['server_host'], dict['job_id'], dict['train_model'],
                   dict['train_model_class_name'],
                   dict['aggregate_strategy'], dict['epoch'], dict['distillation_alpha'], dict['l2_dist'])

#
# class TrainStrategyEncoder(json.JSONEncoder):
#     def default(self, o):
#         if isinstance(o, TrainStrategy):
#             return {
#                 'batch_size': o.get_batch_size(),
#                 # 'fed_strategies': o.get_fed_strategies(),
#                 'loss_function': o.get_loss_function().value,
#                 'optimizer': o.get_optimizer().value
#             }
#         return json.JSONEncoder.default(self, o)
#
#
# class TrainStrategyDecoder(json.JSONDecoder):
#     def decode(self, s, _w=WHITESPACE.match):
#         dict = super().decode(s)
#         # optimizer, learning_rate, loss_function, batch_size, epoch
#         return TrainStrategy(dict['optimizer'], dict['loss_function'],
#                                     dict['batch_size'])


class LoggerFactory(object):

    @staticmethod
    def getLogger(name, level):
        logger = logging.getLogger(name)
        logger.setLevel(level)
        file_handler = logging.FileHandler(LOG_FILE)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        return logger


class CyclicTimer(threading.Timer):
    def run(self):
        while not self.finished.is_set():
            self.function(*self.args, **self.kwargs)
            self.finished.wait(self.interval)

    def cancel(self):
        self.finshed.set()



def return_data_decorator(func):
    def wrapper(*args, **kwargs):
        data, code = func(*args, **kwargs)
        return json.dumps({'data': data, 'code': code})

    return wrapper
