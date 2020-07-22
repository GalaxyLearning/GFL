import os

RUNTIME_CONFIG_SERVER_PATH = os.path.join(os.path.abspath("."), "runtime_config_server.json")
RUNTIME_CONFIG_CLIENT_PATH = os.path.join(os.path.abspath("."), "runtime_config_server.json")
AGGREGATE_PATH = "tmp_aggregate_pars"
JOB_CLIENT_DIR_PATH = os.path.join(os.path.abspath("."), "res", "jobs_client")
JOB_SERVER_DIR_PATH = os.path.join(os.path.abspath("."), "res", "jobs_server")
BASE_MODEL_DIR_PATH = os.path.join(os.path.abspath("."), "res", "models")

g = globals().copy()

for key, value in g.items():
    if key.endswith("_DIR_PATH"):
        os.makedirs(value, exist_ok=True)
class PathFactory(object):

    def __init__(self):
        pass

    @staticmethod
    def get_runtime_config_server_path():
        return RUNTIME_CONFIG_SERVER_PATH

    @staticmethod
    def get_runtime_config_client_path():
        return RUNTIME_CONFIG_CLIENT_PATH

    @staticmethod
    def get_job_client_dir_path():
        return JOB_CLIENT_DIR_PATH

    @staticmethod
    def get_job_server_dir_path():
        return JOB_SERVER_DIR_PATH

    @staticmethod
    def get_base_model_dir_path():
        return BASE_MODEL_DIR_PATH

    @staticmethod
    def get_job_model_path(job_id):
        job_model_path = os.path.join(BASE_MODEL_DIR_PATH, "models_{}".format(job_id))
        os.makedirs(job_model_path, exist_ok=True)
        return job_model_path

    @staticmethod
    def get_job_model_aggregate_path(job_id):
        return os.path.join(PathFactory.get_job_model_path(job_id), "{}".format(AGGREGATE_PATH))

    @staticmethod
    def get_init_model_pars_path(job_id):
        return os.path.join(PathFactory.get_job_model_path(job_id),
                                            "init_model_pars_{}".format(job_id))

    @staticmethod
    def get_first_aggregate_path(job_id):
        dir_path = os.path.join(PathFactory.get_job_model_path(job_id), "tmp_aggregate_pars")
        os.makedirs(dir_path, exist_ok=True)
        return os.path.join(dir_path, "avg_pars_{}".format(0))


    @staticmethod
    def get_job_init_model_code_path(job_id):
        job_model_path = PathFactory.get_job_model_path(job_id)
        return os.path.join(job_model_path, "init_model_{}.py".format(job_id))

    @staticmethod
    def get_model_client_pars_dir(job_id, client_id):

        model_client_pars_dir = os.path.join(PathFactory.get_job_model_path(job_id),
                                                   "models_{}".format(client_id))
        os.makedirs(model_client_pars_dir, exist_ok=True)
        return model_client_pars_dir

    @staticmethod
    def get_model_pars_path(job_id, client_id, fed_step):
        job_model_path = PathFactory.get_job_model_path(job_id)
        dir_path = os.path.join(job_model_path, "models_{}".format(client_id))
        os.makedirs(dir_path, exist_ok=True)
        return os.path.join(dir_path, "tmp_parameters_{}".format(fed_step))

    @staticmethod
    def get_final_model_pars_path(job_id):
        return os.path.join(os.path.abspath("."), "final_model_pars_{}".format(job_id))