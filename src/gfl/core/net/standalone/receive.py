import pickle
from typing import List
import os
from typing import Tuple

import torch

from gfl.core.lfs.path import JobPath
from gfl.core.net.abstract import NetReceive, File
from gfl.net.standalone.cache import get_register_record
from gfl.utils import PathUtils
from gfl.core.scheduler.sql_execute import *


class StandaloneReceive(NetReceive):

    @classmethod
    def receive_job(cls) -> Tuple:
        pass

    @staticmethod
    def get_client_model_paths(job_id, cur_round) -> List[str]:
        """
        获取客户端的模型存储路径
        Returns
        -------

        """
        path_util = JobPath(job_id)
        client_model_paths = []
        client_infos = get_client_by_job_id(job_id)
        for client_info in client_infos:
            client_model_path = path_util.client_params_dir(cur_round, client_info.address) + f"/{job_id}.pth"
            if os.path.exists(client_model_path):
                client_model_paths.append(client_model_path)
                print("聚合方获取训练方的模型：" + str(client_model_path))
        return client_model_paths

    @classmethod
    def receive_partial_params_list(cls, job_id: str, step: int) -> List[dict]:
        partial_params_list = []
        client_model_paths = cls.get_client_model_paths(job_id, step)
        # 加载从client获得的模型参数
        for client_model_path in client_model_paths:
            try:
                client_model_param = torch.load(client_model_path)
                partial_params_list.append(client_model_param)
            except Exception as e:
                raise ValueError(f"模型 {client_model_path} 加载失败"
                                 f"Error: {e}")
        return partial_params_list

    @classmethod
    def receive_partial_params(cls, client_address: str, job_id: str, step: int) -> File:
        """
        获得指定客户端的模型参数
        Parameters
        ----------
        client_address: 客户端的地址
        job_id
        step: 训练任务目前的训练轮数

        Returns
        -------

        """
        client_params_dir = JobPath(job_id).client_params_dir(step, client_address) + f"/{job_id}.pth"
        if os.path.exists(client_params_dir):
            try:
                client_model_param = torch.load(client_params_dir)
            except Exception as e:
                raise ValueError(f"模型 {client_model_param} 加载失败"
                                 f"Error: {e}")
            return torch.load(client_params_dir)
        else:
            return None

    @classmethod
    def receive_global_params(cls, job_id: str, cur_round: int):
        # 在standalone模式下，trainer获取当前聚合轮次下的全局模型
        # 根据 Job 中的 job_id 和 cur_round 获取指定轮次聚合后的 全局模型参数的路径
        global_params_dir = JobPath(job_id).global_params_dir(cur_round)
        model_params_path = PathUtils.join(global_params_dir, job_id + '.pkl')
        # 判断是否存在模型参数文件，如果存在则返回。
        if os.path.exists(global_params_dir) and os.path.isfile(model_params_path):
            # resources_already:1
            # self.__status = JobStatus.RESOURCE_ALREADY
            print("训练方接收全局模型")
            return model_params_path
        else:
            # 等待一段时间。在这段时间内获取到了模型参数文件，则返回
            # 暂时不考虑这种情况
            # 否则，认为当前模型参数文件已经无法获取
            return None

    @classmethod
    def receive_cmd_register(cls, job_id: str):
        return get_register_record(job_id)

    @classmethod
    def receive(cls, client_address: str, job_id: str, step: int, name: str):
        client_dir = JobPath(job_id).client_params_dir(step, client_address)
        client_data_path = client_dir + f"/{name}.pkl"
        if os.path.exists(client_data_path):
            try:
                with open(client_data_path, 'rb') as pkl_file:
                    data = pickle.load(pkl_file)
            except Exception as e:
                raise ValueError(f"数据 {name} 加载失败"
                                 f"Error: {e}")
            return data
        else:
            return None
