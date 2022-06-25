import threading

from gfl.core.scheduler.sql_execute import *

# __register_record = {}
__register_lock = threading.Lock()


def set_register_record(job_id, address, pub_key, dataset_id):
    # global __register_record
    __register_lock.acquire()
    client = ClientEntity(address, pub_key, dataset_id)
    save_client(job_id, client=client)
    # client_list = __register_record.get(job_id)
    # if client_list is None:
    #     client_list = []
    #     __register_record[job_id] = client_list
    # client_list.append({
    #     "address": address,
    #     "pub_key": pub_key,
    #     "dataset_id": dataset_id
    # })
    __register_lock.release()


def get_register_record(job_id):
    __register_lock.acquire()
    clients = get_client_by_job_id(job_id=job_id)
    # client_list = __register_record.get(job_id, None)
    # if client_list:
    #     client = client_list.pop(0)
    # else:
    #     client = None
    __register_lock.release()
    return clients
