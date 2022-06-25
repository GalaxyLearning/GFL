import os
import pickle
import sys
from typing import NoReturn

import torch

from gfl.core.lfs.path import JobPath
from gfl.core.lfs.path import JobPath
from gfl.core.net.abstract import NetSend, File
from gfl.net.standalone.cache import set_register_record
from gfl.utils import PathUtils


class StandaloneSend(NetSend):

    @classmethod
    def send_partial_params(cls, client: str, job_id: str, step: int, params) -> NoReturn:
        # 这里的参数client，暂时认为是client_address
        # 在standalone模式下，trainer当前训练轮次得到的模型保存在指定路径下
        client_params_dir = JobPath(job_id).client_params_dir(step, client)
        os.makedirs(client_params_dir, exist_ok=True)
        # 保存 job_id.pth为文件名
        path = PathUtils.join(client_params_dir, job_id + '.pkl')
        # path = client_params_dir + 'job_id.pth'
        # torch.save(params, path)
        with open(path, 'wb') as f:
            pickle.dump(params, f)
        print("训练完成，已将模型保存至：" + str(client_params_dir))

    @classmethod
    def send_cmd_register(cls, job_id: str, address: str, pub_key: str, dataset_id: str):
        set_register_record(job_id, address, pub_key, dataset_id)

    @classmethod
    def send(cls, client_address: str, job_id: str, step: int, name: str, data):
        client_dir = JobPath(job_id).client_params_dir(step, client_address)
        client_data_path = client_dir + f"/{name}.pkl"
        try:
            with open(client_data_path, 'wb') as pkl_file:
                pickled_data = pickle.dumps(data)
                data_size = sys.getsizeof(pickled_data)
                pkl_file.write(pickled_data)
        except Exception as e:
            raise ValueError(f"数据 {data} 发送失败"
                             f"Error: {e}")
        return data_size
