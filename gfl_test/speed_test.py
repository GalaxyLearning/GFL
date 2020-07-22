import os
import time

from gfl_test.utils import running_time

DIR = "data"


"""
exec 10000000 times
clock time: 0.49294399999999994
real time: 0.4652895927429199
"""
@running_time(exec_times=10000000)
def get_path():
    return "data/t1"


"""
exec 10000000 times
clock time: 0.48937299999999995
real time: 0.511483907699585
"""
@running_time(exec_times=10000000)
def get_add_path():
    return "data/" + "t1"


"""
exec 10000000 times
clock time: 7.236731000000001
real time: 7.352431774139404
"""
@running_time(exec_times=10000000)
def get_join_path():
    return os.path.join(DIR, "t1")


"""
exec 10000000 times
clock time: 14.042800999999999
real time: 14.201583862304688
"""
@running_time(exec_times=10000000)
def get_create_path():
    if os.path.exists("data/t2"):
        os.makedirs("data/t2", exist_ok=True)
    return "data/t2"


if __name__ == "__main__":
    get_path()
    get_add_path()
    get_join_path()
    get_create_path()

