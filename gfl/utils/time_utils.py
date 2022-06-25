import time


class TimeUtils(object):

    @classmethod
    def millis_time(cls):
        return int(1000 * time.time())
