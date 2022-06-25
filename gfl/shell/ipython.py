import os

import IPython
from traitlets.config import Config


def startup():
    c = Config()

    c.InteractiveShellApp.exec_lines = [
        "from gfl.shell.ipython_startup import gfl, node"
    ]
    c.InteractiveShell.colors = 'Neutral'
    c.InteractiveShell.confirm_exit = False
    c.TerminalIPythonApp.display_banner = False

    # Now we start ipython with our configuration
    IPython.start_ipython(argv=[], config=c)
