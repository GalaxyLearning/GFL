import os
import sys

import requests

from gfl.conf import GflConf


class Shell(object):

    __host = "127.0.0.1"
    __port = 9434

    @classmethod
    def welcome(cls, **kwargs):
        print("------- GFL -------")
        print("%-20s:%s" % ("pid", str(os.getpid())))

    @classmethod
    def attach(cls, host, port):
        cls.welcome()
        cls.startup(host=host, port=port)
        pass

    @classmethod
    def startup(cls, **kwargs):
        cls.__host = kwargs.pop("host", "127.0.0.1")
        cls.__port = kwargs.pop("port", GflConf.get_property("api.http.port"))
        while True:
            cmd = input("> ")
            if "EXIT".lower() == cmd.lower():
                cls.exit()
                break
            if cmd.startswith("SHOWCONF"):
                key = cmd[9:].strip()
                print(GflConf.get_property(key))

    @classmethod
    def exit(cls, **kwargs):
        req_url = "http://%s:%s/shutdown" % (cls.__host, cls.__port)
        resp = requests.post(req_url)
        try:
            data = resp.json()
            if data["code"] == 0:
                return True
            else:
                return False
        except:
            return False
