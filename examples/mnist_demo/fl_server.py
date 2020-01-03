from pfl.core.server import FlStandaloneServer
from pfl.core.strategy import FederateStrategy

FEDERATE_STRATEGY = FederateStrategy.FED_AVG

if __name__ == "__main__":

    FlStandaloneServer(FEDERATE_STRATEGY).start()
