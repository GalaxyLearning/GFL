from pfl.core.server import FlClusterServer
from pfl.core.strategy import FederateStrategy

FEDERATE_STRATEGY = FederateStrategy.FED_DISTILLATION
IP = '0.0.0.0'
PORT = 9763
API_VERSION = '/api/version'

if __name__ == "__main__":

    FlClusterServer(FEDERATE_STRATEGY, IP, PORT, API_VERSION).start()
