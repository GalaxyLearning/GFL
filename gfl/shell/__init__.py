import os

from .ipython import startup as ipython_startup


def startup(shell_type, home=None, node_ip=None, node_port=None):
    if home is not None:
        os.environ["__GFL_HOME__"] = home
    if node_ip is not None:
        os.environ["__GFL_NODE_IP__"] = node_ip
    if node_port is not None:
        os.environ["__GFL_NODE_PORT__"] = node_port

    if shell_type is None or not isinstance(shell_type, str):
        raise ValueError(f"shell_type should be a str")
    if shell_type.lower() == "ipython":
        ipython_startup()
    else:
        raise ValueError(f"{shell_type} is not supported")
