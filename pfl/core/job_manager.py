import torch
import threading
import pickle
import os, json
import inspect
from pfl.entity import runtime_config
from pfl.entity.job import Job
from pfl.utils.utils import JobUtils
from pfl.core.strategy import WorkModeStrategy

lock = threading.RLock()


class JobManager(object):

    def __init__(self, job_path):
        self.job_path = job_path

    def generate_job(self, work_mode, train_code_strategy, fed_strategy, model, distillation_alpha):
        with lock:
            # server_host, job_id, train_strategy, train_model, train_model_class_name, fed_strategy, iterations, distillation_alpha
            job = Job(None, JobUtils.generate_job_id(), train_code_strategy, inspect.getsourcefile(model),
                      model.__name__, fed_strategy, distillation_alpha)
            if work_mode == WorkModeStrategy.WORKMODE_STANDALONE:
                job.set_server_host("localhost:8080")
            else:
                job.set_server_host("")

            return job

    def submit_job(self, job, model, model_path):

        with lock:
            # create model dir of this job
            job_model_dir = os.path.join(model_path, "models_{}".format(job.get_job_id()))
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

            print("job {} added successfully".format(job.get_job_id()))

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
