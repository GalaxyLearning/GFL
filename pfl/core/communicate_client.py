import os, logging
from flask import Flask, request
from werkzeug.serving import run_simple
from pfl.utils.utils import return_data_decorator, LoggerFactory

app = Flask(__name__)

BASE_MODEL_PATH = os.path.join(os.path.abspath("."), "res", "models")

logger = LoggerFactory.getLogger(__name__, logging.INFO)


@return_data_decorator
@app.route("/", methods=['GET'])
def test_client():
    return "Hello pfl client", 200


@return_data_decorator
@app.route("/aggregatepars", methods=['POST'])
def submit_aggregate_pars():
    logger.info("receive aggregate files")
    recv_aggregate_files = request.files
    # print(recv_aggregate_files)
    for filename in recv_aggregate_files:
        job_id = filename.split("_")[-2]
        # print("recv_filename: ", recv_aggregate_files[filename])
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