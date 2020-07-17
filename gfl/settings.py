import os
import tempfile


geth_url = ""
ipfs_url = ""

DATA_DIR_PATH = "data"
TEMP_DIR_PATH = os.path.join(tempfile.gettempdir(), "gfl")
RUNTIME_CONFIG_SERVER_PATH = os.path.join(os.path.abspath("."), "runtime_config_server.json")
RUNTIME_CONFIG_CLIENT_PATH = os.path.join(os.path.abspath("."), "runtime_config_server.json")
JOB_CLIENT_DIR_PATH = os.path.join(os.path.abspath("."), "res", "jobs_client")
JOB_SERVER_DIR_PATH = os.path.join(os.path.abspath("."), "res", "jobs_server")
BASE_MODEL_DIR_PATH = os.path.join(os.path.abspath("."), "res", "models")

g = globals()

for key, value in g.items():
    if key.rfind("_DIR_PATH"):
        os.makedirs(value, exist_ok=True)

