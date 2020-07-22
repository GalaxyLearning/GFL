from functools import wraps
import time


def running_time(exec_times=1):

    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            start_clock = time.clock()
            start_time = time.time()
            ret = None
            for i in range(0, exec_times):
                ret = func(*args, *kwargs)
            end_clock = time.clock()
            end_time = time.time()
            print("Exec %s for %s times." % (func.__name__, exec_times))
            print(" |--clock time: %sms" % (1000 * (end_clock - start_clock)))
            print(" |--real  time: %sms" % (1000 * (end_time - start_time)))
            return ret

        return wrapper

    return decorator