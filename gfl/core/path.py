import os

from gfl.settings import DATA_DIR_PATH


JOB_CLIENT_DIR_PATH = os.path.join(DATA_DIR_PATH, "res", "jobs_client")
JOB_SERVER_DIR_PATH = os.path.join(DATA_DIR_PATH, "res", "jobs_server")
BASE_MODEL_DIR_PATH = os.path.join(DATA_DIR_PATH, "res", "models")


class JobPath(object):

    @classmethod
    def partial_params_path(cls, job_id, client_id):
        pass

    @classmethod
    def aggrefate_params_path(cls, job_id, fed_step):
        pass

    @classmethod
    def model_path(cls, job_id):
        pass


