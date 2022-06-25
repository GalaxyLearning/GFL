import os
import pickle
from typing import NoReturn

from gfl.core.lfs.path import JobPath
from gfl.core.net.abstract import NetBroadcast, File


class StandaloneBroadcast(NetBroadcast):

    @classmethod
    def broadcast_job(cls, job_id: str, job: File) -> NoReturn:
        # Nothing to do
        pass

    @classmethod
    def broadcast_dataset(cls, dataset_id: str, dataset: File) -> NoReturn:
        # Nothing to do
        pass

    # @classmethod
    # def broadcast_global_params(cls, job_id: str, step: int, params) -> NoReturn:
    #     # 在 standalone 模式下，将聚合后的模型保存到指定位置
    #     global_params_path = JobPath(job_id).global_params_dir(step)
    #     os.makedirs(global_params_path, exist_ok=True)
    #     path = PathUtils.join(global_params_path, job_id + '.pth')
    #     # 将聚合后的模型参数保存在指定路径上
    #     torch.save(params, path)
    #     print("聚合完成，已经模型保存至：" + str(global_params_path))

    @classmethod
    def broadcast(cls, job_id: str, step: int, data, name):
        path_util = JobPath(job_id)
        global_path = path_util.global_params_dir(step)
        os.makedirs(global_path, exist_ok=True)
        data_path = global_path + f"/{name}.pkl"
        with open(data_path, "wb") as f:
            pickle.dump(data, f)
        print("聚合完成，已经模型保存至：" + str(global_path))
