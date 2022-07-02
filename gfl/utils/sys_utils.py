
import psutil


class SysUtils(object):

    @classmethod
    def cpu_count(cls, logical=True):
        return psutil.cpu_count(logical)

    @classmethod
    def cpu_percent(cls, index=None):
        if index is None or index < 0:
            return psutil.cpu_percent()
        else:
            return psutil.cpu_percent(percpu=True)[index]

    @classmethod
    def mem_total(cls):
        pass

    @classmethod
    def mem_used(cls):
        pass

    @classmethod
    def mem_available(cls):
        pass

    @classmethod
    def mem_free(cls):
        pass

    @classmethod
    def gpu_count(cls):
        pass

    @classmethod
    def gpu_mem_total(cls, index):
        pass

    @classmethod
    def gpu_mem_used(cls, index):
        pass

    @classmethod
    def gpu_mem_free(cls, index):
        pass

    @classmethod
    def gpu_utilization_rates(cls, index):
        pass

    @classmethod
    def proc_cpu_percent(cls, pid=None):
        pass

    @classmethod
    def proc_mem_used(cls, pid=None):
        pass

    @classmethod
    def proc_gpu_mem_used(cls, pid=None, index=None):
        pass
