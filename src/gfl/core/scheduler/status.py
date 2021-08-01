from enum import Enum


class JobStatus(Enum):
    """
    标识任务的状态
    """
    RESOURCE_NOT_ALREADY = 0    #:
    RESOURCE_ALREADY = 1        #:
    TRAINING = 2
    EPOCH_FINISHED = 3
    ALL_FINISHED = 4
    TRAIN_FAILED = 5
