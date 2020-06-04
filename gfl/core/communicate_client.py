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

import os, logging
from flask import Flask, request
from werkzeug.serving import run_simple
from gfl.utils.utils import return_data_decorator, LoggerFactory

app = Flask(__name__)

BASE_MODEL_PATH = os.path.join(os.path.abspath("."), "res", "models")

logger = LoggerFactory.getLogger(__name__, logging.INFO)


@return_data_decorator
@app.route("/", methods=['GET'])
def test_client():
    return "Hello gfl client", 200


@return_data_decorator
@app.route("/aggregatepars", methods=['POST'])
def submit_aggregate_pars():
    logger.info("receive aggregate files")
    recv_aggregate_files = request.files

    for filename in recv_aggregate_files:
        job_id = filename.split("_")[-2]
        tmp_aggregate_file = recv_aggregate_files[filename]
        job_base_model_dir = os.path.join(BASE_MODEL_PATH, "models_{}".format(job_id), "tmp_aggregate_pars")
        latest_num = len(os.listdir(job_base_model_dir)) - 1
        latest_tmp_aggretate_file_path = os.path.join(job_base_model_dir, "avg_pars_{}".format(latest_num))
        with open(latest_tmp_aggretate_file_path, "wb") as f:
            for line in tmp_aggregate_file.readlines():
                f.write(line)
        logger.info("recv success")
    return "ok", 200


def start_communicate_client(client_ip, client_port):
    app.url_map.strict_slashes = False
    run_simple(hostname=client_ip, port=int(client_port), application=app, threaded=True)
    logger.info("galaxy learning client started")