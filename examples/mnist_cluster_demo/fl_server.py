from gfl.core.server import FLClusterServer

IP = '0.0.0.0'
PORT = 9763
API_VERSION = '/api/version'

if __name__ == "__main__":

    FLClusterServer(IP, PORT, API_VERSION).start()
