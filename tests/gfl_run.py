import os

from gfl.application import Application
from gfl import GflConf


GflConf.home_dir = os.path.join(os.path.dirname(__file__), "../data")


if __name__ == "__main__":
    Application.run(True)
