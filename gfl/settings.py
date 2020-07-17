import os

RUNTIME_CONFIG_SERVER_PATH = os.path.join(os.path.abspath("."), "runtime_config_server.json")
RUNTIME_CONFIG_CLIENT_PATH = os.path.join(os.path.abspath("."), "runtime_config_server.json")
JOB_CLIENT_DIR_PATH = os.path.join(os.path.abspath("."), "res", "jobs_client")
JOB_SERVER_DIR_PATH = os.path.join(os.path.abspath("."), "res", "jobs_server")
BASE_MODEL_DIR_PATH = os.path.join(os.path.abspath("."), "res", "models")

g = globals()

for key, value in g.items():
    if key.rfind("_DIR_PATH"):
        if not os.path.exists(value):
            os.mkdir(value)
