import os
import tempfile


geth_url = "http://127.0.0.1:8545"
ipfs_url = "/dns/dev.malanore.cn/tcp/9013/http"

DATA_DIR_PATH = "data"
TEMP_DIR_PATH = os.path.join(tempfile.gettempdir(), "gfl")
RUNTIME_CONFIG_SERVER_PATH = os.path.join(os.path.abspath("."), "runtime_config_server.json")
RUNTIME_CONFIG_CLIENT_PATH = os.path.join(os.path.abspath("."), "runtime_config_server.json")
JOB_CLIENT_DIR_PATH = os.path.join(os.path.abspath("."), "res", "jobs_client")
JOB_SERVER_DIR_PATH = os.path.join(os.path.abspath("."), "res", "jobs_server")
BASE_MODEL_DIR_PATH = os.path.join(os.path.abspath("."), "res", "models")

g = globals().copy()

for key, value in g.items():
    if key.endswith("_DIR_PATH"):
        os.makedirs(value, exist_ok=True)

