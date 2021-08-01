import inspect
import json
from io import BytesIO

from gfl.core.data import Job, Dataset
from gfl.utils import ModuleUtils, ZipUtils


def pack_job(job: Job):
    data = {
        "job_id": job.job_id,
        "metadata": job.metadata.to_dict(),
        "job_config": job.job_config.to_dict(),
        "train_config": job.train_config.to_dict(),
        "aggregate_config": job.aggregate_config.to_dict()
    }
    module_path = ModuleUtils.get_module_path(job.module)
    module_zip_data = ZipUtils.get_compress_data(module_path, job.job_id)
    return json.dumps(data).encode("utf-8"), module_zip_data


def pack_dataset(dataset: Dataset):
    data = {
        "dataset_id": dataset.dataset_id,
        "metadata": dataset.metadata,
        "dataset_config": dataset.dataset_config
    }
    module_path = ModuleUtils.get_module_path(dataset.module)
    module_zip_data = ZipUtils.get_compress_data(module_path, dataset.dataset_id)
    return json.dumps(data).encode("utf-8"), module_zip_data
