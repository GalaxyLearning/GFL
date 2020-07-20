import os

from gfl.settings import DATA_DIR_PATH, TEMP_DIR_PATH


"""
RUNTIME_CONFIG_SERVER_PATH = os.path.join(os.path.abspath("."), "runtime_config_server.json")
RUNTIME_CONFIG_CLIENT_PATH = os.path.join(os.path.abspath("."), "runtime_config_server.json")
JOB_CLIENT_DIR_PATH = os.path.join(os.path.abspath("."), "res", "jobs_client")
JOB_SERVER_DIR_PATH = os.path.join(os.path.abspath("."), "res", "jobs_server")
BASE_MODEL_DIR_PATH = os.path.join(os.path.abspath("."), "res", "models")
"""


def __create_or_get_path(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


def client_dir_path():
    return __create_or_get_path(os.path.join(DATA_DIR_PATH, "res", "job_client"))


def server_dir_path():
    return __create_or_get_path(os.path.join(DATA_DIR_PATH, "res", "job_server"))


def model_dir_path():
    return __create_or_get_path(os.path.join(DATA_DIR_PATH, "res", "models"))


def partial_params_path(job_id, client_id):
    pass


def aggregate_params_path(job_id):
    pass


def model_path(job_id):
    pass