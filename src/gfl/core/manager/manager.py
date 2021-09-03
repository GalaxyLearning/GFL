import threading

import gfl.core.lfs as lfs
from gfl.core.manager.node import GflNode
from gfl.core.lfs import Lfs


# 发布job和运行job的过程：
# 1、任意一个节点都可以新建一个job,按序进行初始化、广播、运行直至job达到结束条件
#   1.1 新建一个job，设置模型、数据、聚合参数、训练参数等，并且需要通过配置文件的方式（也可以调用相关的方法）生成与这个job相关联的topology_manager
#   1.2 调用init_job_sqlite、submit_job完成初始化操作，并且进行广播，并在数据库中甚至为waiting状态，等待接下来运行
#   1.3 运行job（何时创建scheduler？）
# 2、其余节点监听到此job之后，进行初始化操作。在运行这个job获取job和topology_manager。根据这个job与本节点是否有关，保存在本地。并创建对应的scheduler
class NodeManager(object):

    """

    """

    def __init__(self, *, node: GflNode = None, role: str = "client"):
        super(NodeManager, self).__init__()
        self.node = GflNode.default_node if node is None else node
        self.role = role
        self.waiting_list = None
        self.scheduler_list = []
        self._is_stop = False

    __instance = None
    __instance_lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        if cls.__instance is None:
            with cls.__instance_lock:
                if cls.__instance is None:
                    cls.__instance = NodeManager()
        return cls.__instance

    def run(self):
        while True:
            if self._is_stop is True:
                break

            pass
        # release resources

    def get_relative_jobs(self):
        pass

    def get_all_jobs(self):
        pass

    def get_job(self, job_id: str):
        pass

    def sync_job(self, job_id: str):
        pass

    def listen_job(self):
        # 监听job，并将监听到的job保存到waiting_list中
        # 在单机模式下，所有节点共用一个数据库，所以直接调用此方法获取未完成的job
        jobs = JobManager.unfinished_jobs()
        self.waiting_list = jobs
        # 在多机模式下，各个节点都有各自的数据库
        # 1、调用通信模块的方法监听其余节点发送过来的job
        # 2、将其存储到该节点的数据库当中
        # 3、jobs = JobManager.unfinished_jobs()
        # 4、self.waiting_list = jobs

    def stop(self):
        self._is_stop = True
