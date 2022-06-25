import logging
import os
import shutil
import sys

from daemoniker import Daemonizer

from gfl.api.listener import HttpListener
from gfl.conf import GflConf
from gfl.core.lfs import Lfs
from gfl.core.manager import GflNode, NodeManager
from gfl.shell import Shell
from gfl.utils import PathUtils


class Application(object):

    logger = None

    def __init__(self):
        super(Application, self).__init__()

    @classmethod
    def init(cls, force):
        if os.path.exists(GflConf.home_dir):
            if force:
                logging.shutdown()
                shutil.rmtree(GflConf.home_dir)
            else:
                raise ValueError("homedir not empty.")
        # create home dir
        os.makedirs(GflConf.home_dir)

        # generate config file
        GflConf.generate_config(PathUtils.join(GflConf.home_dir, "gfl_config.yaml"))
        # generate node address and key
        GflNode.init_node()
        # create data directories
        Lfs.init()

    @classmethod
    def run(cls, console=True, **kwargs):
        sys.stderr = open(os.devnull, "w")
        cls.logger = logging.getLogger("gfl_p")
        with Daemonizer() as (is_setup, daemonizer):
            main_pid = None
            if is_setup:
                main_pid = os.getpid()
            pid_file = PathUtils.join(GflConf.home_dir, "proc.lock")
            stdout_file = PathUtils.join(GflConf.logs_dir, "console_out")
            stderr_file = PathUtils.join(GflConf.logs_dir, "console_err")
            is_parent = daemonizer(pid_file, stdout_goto=stdout_file, stderr_goto=stderr_file)
            if is_parent:
                if console and main_pid == os.getpid():
                    Shell.startup()

        GflNode.load_node()

        HttpListener.start()

        NodeManager.get_instance().run()
