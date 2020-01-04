from pfl.core.server import FLStandaloneServer
from pfl.core.strategy import FederateStrategy

FEDERATE_STRATEGY = FederateStrategy.FED_AVG

if __name__ == "__main__":

    FLStandaloneServer(FEDERATE_STRATEGY).start()
