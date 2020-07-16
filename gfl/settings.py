import os
import tempfile


geth_url = ""
ipfs_url = ""

data_dir = "data"
temp_dir = os.path.join(tempfile.gettempdir(), "gfl")

os.makedirs(data_dir, exist_ok=True)