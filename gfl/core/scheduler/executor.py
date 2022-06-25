import abc

from gfl.data import Job
from gfl.core.scheduler.status import JobStatus


class JobExecutor(object):

    def __init__(self, *, job: Job, step: int):
        super(JobExecutor, self).__init__()
        self.job = job
        self.step = step

    @abc.abstractmethod
    def start(self):
        pass

    @abc.abstractmethod
    def status(self):
        pass

    @abc.abstractmethod
    def stop(self):
        pass

    @abc.abstractmethod
    def is_available(self):
        pass

    @abc.abstractmethod
    def is_running(self):
        pass

    @abc.abstractmethod
    def is_finished(self):
        pass


class JobTrainExecutor(JobExecutor):

    def __init__(self, *, job: Job, step: int):
        super(JobTrainExecutor, self).__init__(job=job, step=step)
        self.trainer = None
        self.step = step
        self.__status = JobStatus.RESOURCE_NOT_ALREADY

    def init_trainer(self):
        self.job.job_config.trainer.is_instance = True
        trainer_clazz = self.job.job_config.get_trainer()


    def make_dirs(self):
        cur_round = self.job.cur_round

    def train(self):
        pass

    def start(self):
        """

        :return:
        """
        self.make_dirs()
        self.train()
        self.job.cur_round += 1
        return self.is_finished()

    def status(self):
        return self.__status

    def stop(self):
        pass

    def is_available(self):
        pass

    def is_running(self):
        pass

    def is_finished(self):
        pass


class JobAggregateExecutor(JobExecutor):

    def __init__(self, *, job: Job, step: int):
        super(JobAggregateExecutor, self).__init__(job=job, step=step)

    def start(self):
        pass

    def status(self):
        pass

    def stop(self):
        pass

    def is_available(self):
        pass

    def is_running(self):
        pass

    def is_finished(self):
        pass
