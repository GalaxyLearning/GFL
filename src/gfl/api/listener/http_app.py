__all__ = [
    "HttpListener"
]

import json
import threading

from flask import Flask, request

from gfl.conf import GflConf
from gfl.core.data import *
from gfl.core.lfs import Lfs
from gfl.core.manager import NodeManager


app = Flask(__name__)


@app.route("/test", methods=["GET", "POST"])
def app_test():
    if request.method == "GET":
        return {
            "method": "GET",
            "args": dict(request.args)
        }
    else:
        return {
            "method": "POST",
            "data": dict(request.form)
        }


@app.route("/connected")
def is_connected():
    print("connected entry")
    return {
        "code": 0
    }


@app.route("/shutdown", methods=["POST"])
def shutdown():
    if request.method == "POST":
        NodeManager.get_instance().stop()
        return {
            "code": 0
        }


@app.route("/submit", methods=["POST"])
def submit():
    if request.method == "POST":
        if request.args["type"] == "job":
            job_data = request.files["job_data"].stream.read()
            module_data = request.files["module_data"].stream.read()
            data_dict = json.loads(job_data.decode("utf-8"))
            job = Job(job_id=data_dict["job_id"],
                      metadata=data_dict["metadata"],
                      job_config=data_dict["job_config"],
                      train_config=data_dict["train_config"],
                      aggregate_config=data_dict["aggregate_config"])
            Lfs.save_job(job, module_data=module_data)
            return {
                "code": 0
            }
        elif request.args["type"] == "dataset":
            dataset_data = request.files["dataset_data"].stream.read()
            module_data = request.files["module_data"].stream.read()
            data_dict = json.loads(dataset_data.decode("utf-8"))
            dataset = Dataset(dataset_id=data_dict["dataset_id"],
                              metadata=data_dict["metadata"],
                              dataset_config=data_dict["dataset_config"])
            Lfs.save_dataset(dataset, module_data=module_data)
            return {
                "code": 0
            }
        else:
            return {
                "code": 1,
                "errmsg": "unknown type"
            }


class HttpListener(object):

    @classmethod
    def run(cls) -> None:
        bind_ip = GflConf.get_property("api.http.bind_ip")
        port = int(GflConf.get_property("api.http.port"))
        app.logger.disabled = True
        app.run(host=bind_ip, port=port)

    @classmethod
    def start(cls) -> None:
        t = threading.Thread(target=cls.run)
        t.setDaemon(True)
        t.start()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=9434, debug=True)
