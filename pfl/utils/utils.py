import os
import json
import threading
import datetime
import pickle
import requests
import logging
from json.decoder import WHITESPACE

from pfl.entity.job import Job
from pfl.core.strategy import TrainStrategyFatorcy

LOG_FILE = os.path.join(os.path.abspath("."), "log.txt")


class JobIdCount(object):
    lock = threading.RLock()

    def __init__(self, init_value):
        self.value = init_value

    def incr(self, step):
        with JobIdCount.lock:
            self.value += step
            return self.value


class Utils(object):
    def __init__(self):
        pass


jobCount = JobIdCount(init_value=0)


class JobUtils(Utils):
    def __init__(self):
        super(JobUtils, self).__init__()

    @staticmethod
    def generate_job_id():
        return '{}{}'.format(datetime.datetime.now().strftime("%Y%m%d%H%M%S%f"), jobCount.incr(1))

    @staticmethod
    def list_all_jobs(job_path, job_iter_dict):
        job_list = []
        for file in os.listdir(job_path):
            # print("job file: ", job_path+"\\"+file)
            with open(os.path.join(job_path, file), "rb") as f:
                job = pickle.load(f)
                job_list.append(job)
                if job_iter_dict.get(job.get_job_id()) is None:
                    job_iter_dict[job.get_job_id()] = 0
        return job_list

    @staticmethod
    def serialize(job):
        return pickle.dumps(job)

    @staticmethod
    def get_job_from_remote(server_url, job_path):
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


class JobEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Job):
            return {
                'job_id': o.get_job_id(),
                'train_model': o.get_train_model(),
                'train_strategy': json.dumps(o.get_train_strategy(), cls=TrainStrategyFatorcyEncoder),
                'train_model_class_name': o.get_train_model_class_name(),
                'server_host': o.get_server_host(),
                'aggregate_strategy': o.get_aggregate_strategy().value,
                'distillation_alpha': o.get_distillation_alpha()
            }
        return json.JSONEncoder.default(self, o)


class JobDecoder(json.JSONDecoder):
    def decode(self, s, _w=WHITESPACE.match):
        dict = super().decode(s)
        # server_host, job_id, train_strategy, train_model, train_model_class_name, iterations
        return Job(dict['server_host'], dict['job_id'],
                   json.loads(dict['train_strategy'], cls=TrainStrategyFactoryDecoder), dict['train_model'],
                   dict['train_model_class_name'],
                   dict['aggregate_strategy'], dict['distillation_alpha'])


class TrainStrategyFatorcyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, TrainStrategyFatorcy):
            return {
                'batch_size': o.get_batch_size(),
                'epoch': o.get_epoch(),
                # 'fed_strategies': o.get_fed_strategies(),
                'learning_rate': o.get_learning_rate(),
                'loss_function': o.get_loss_function().value,
                'optimizer': o.get_optimizer().value
            }
        return json.JSONEncoder.default(self, o)


class TrainStrategyFactoryDecoder(json.JSONDecoder):
    def decode(self, s, _w=WHITESPACE.match):
        dict = super().decode(s)
        # optimizer, learning_rate, loss_function, batch_size, epoch
        return TrainStrategyFatorcy(dict['optimizer'], dict['learning_rate'], dict['loss_function'],
                                    dict['batch_size'], dict['epoch'])


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


def return_data_decorator(func):
    def wrapper(*args, **kwargs):
        data, code = func(*args, **kwargs)
        return json.dumps({'data': data, 'code': code})

    return wrapper
