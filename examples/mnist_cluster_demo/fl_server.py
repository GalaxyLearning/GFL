from pfl.core.server import FlStandaloneServer, FlClusterServer
from pfl.core.strategy import WorkModeStrategy, FederateStrategy

WORK_MODE = WorkModeStrategy.WORKMODE_CLUSTER
FEDERATE_STRATEGY = FederateStrategy.FED_DISTILLATION
IP = '0.0.0.0'
PORT = 9763
API_VERSION = '/api/version'

if __name__ == "__main__":

    if WORK_MODE == WorkModeStrategy.WORKMODE_STANDALONE:
        FlStandaloneServer(FEDERATE_STRATEGY).start()
    else:
        FlClusterServer(FEDERATE_STRATEGY, IP, PORT, API_VERSION).start()
