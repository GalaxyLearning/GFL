import threading
import time
import os
from concurrent.futures import ThreadPoolExecutor
from pfl.core.aggregator import FedAvgAggregator
from pfl.core.strategy import WorkModeStrategy, FederateStrategy
from pfl.core import communicate_server

JOB_PATH = os.path.join(os.path.abspath("."), "res", "jobs_server")
BASE_MODEL_PATH = os.path.join(os.path.abspath("."), "res", "models")


class FlServer(object):

    def __init__(self):
        super(FlServer, self).__init__()


class FlStandaloneServer(FlServer):
    def __init__(self, federate_strategy):
        super(FlStandaloneServer, self).__init__()
        self.executor_pool = ThreadPoolExecutor(5)
        if federate_strategy == FederateStrategy.FED_AVG:
            self.aggregator = FedAvgAggregator(WorkModeStrategy.WORKMODE_STANDALONE, JOB_PATH, BASE_MODEL_PATH)
        else:
            pass

    def start(self):
        self.executor_pool.submit(self.aggregator.aggregate)
        #self.aggregator.aggregate()


class FlClusterServer(FlServer):

    def __init__(self, federate_strategy, ip, port, api_version):
        super(FlClusterServer, self).__init__()
        self.executor_pool = ThreadPoolExecutor(5)
        if federate_strategy == FederateStrategy.FED_AVG:
            self.aggregator = FedAvgAggregator(WorkModeStrategy.WORKMODE_CLUSTER, JOB_PATH, BASE_MODEL_PATH)
        else:
            pass
        self.ip = ip
        self.port = port
        self.api_version = api_version
        self.federate_strategy = federate_strategy

    def start(self):
        self.executor_pool.submit(communicate_server.start_communicate_server, self.api_version, self.ip, self.port)
        # self.executor_pool.submit(self.aggregator.aggregate)
        # communicate_server.start_communicate_server(self.api_version, self.ip, self.port)
        if self.federate_strategy == FederateStrategy.FED_AVG:
            self.aggregator.aggregate()
        else:
            pass
