from gfl.core.server import FLClusterServer
from gfl.core.strategy import FederateStrategy

FEDERATE_STRATEGY = FederateStrategy.FED_AVG
IP = '0.0.0.0'
PORT = 9763
API_VERSION = '/api/version'

if __name__ == "__main__":

    FLClusterServer(FEDERATE_STRATEGY, IP, PORT, API_VERSION).start()
