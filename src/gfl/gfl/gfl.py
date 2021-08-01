import json

import requests

from gfl.core.data import Job, Dataset
from gfl.gfl.pack import pack_job, pack_dataset


class GFL(object):

    def __init__(self, host="127.0.0.1", port=9434):
        super(GFL, self).__init__()
        self.host = host
        self.port = port

    @property
    def url(self):
        return "http://%s:%s" % (self.host, self.port)

    def is_connected(self):
        req_url = self.__concat_url("connected")
        try:
            resp = requests.get(req_url)
            data = resp.json()
            if data["code"] == 0:
                return True
            else:
                return False
        except:
            return False

    def submit_job(self, job: Job):
        job_data, module_data = pack_job(job)
        req_url = self.__concat_url("submit", {"type": "job"})
        try:
            resp = requests.post(req_url, files={
                "job_data": job_data,
                "module_data": module_data
            })
            resp_data = resp.json()
            if resp_data["code"] == 0:
                return True
            else:
                return False
        except:
            return False

    def submit_dataset(self, dataset: Dataset):
        dataset_data, module_data = pack_dataset(dataset)
        req_url = self.__concat_url("submit", {"type": "dataset"})
        try:
            resp = requests.post(req_url, files={
                "dataset_data": dataset_data,
                "module_data": module_data
            })
            resp_data = resp.json()
            if resp_data["code"] == 0:
                return True
            else:
                return False
        except:
            return False

    def __concat_url(self, path, params=None):
        if params is None:
            params = {}
        params_str = "&".join(["%s=%s" % (k, v) for k, v in params.items()])
        return "%s/%s?%s" % (self.url, path, params_str)
