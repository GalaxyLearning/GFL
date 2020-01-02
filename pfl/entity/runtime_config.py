WAITING_JOB_LIST = list()
PENDING_JOB_LIST = list()
EXEC_JOB_LIST = list()
CONNECTED_TRAINER_LIST = list()
WAITING_BROADCAST_AGGREGATED_JOB_ID_LIST = list()


def get_waiting_job():
    return WAITING_JOB_LIST


def add_waiting_job(job):
    WAITING_JOB_LIST.append(job)


def remove_waiting_job(job):
    WAITING_JOB_LIST.remove(job)


def get_pending_job():
    return PENDING_JOB_LIST


def add_pending_job(job):
    PENDING_JOB_LIST.append(job)


def remove_pending_job(job):
    PENDING_JOB_LIST.remove(job)


def add_exec_job(job):
    EXEC_JOB_LIST.put(job)


def get_exec_job():
    return EXEC_JOB_LIST.get()
