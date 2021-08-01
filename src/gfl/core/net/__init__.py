__all__ = [
    "NetBroadcast",
    "NetFetch",
    "NetReceive",
    "NetSend",
    "NetCallback"
]

import warnings

from gfl.conf import GflConf


net_mode = GflConf.get_property("net.mode")

if net_mode == "standalone":
    from gfl.core.net.standalone import NetBroadcast, NetFetch, NetReceive, NetSend, NetCallback
elif net_mode == "http":
    from gfl.core.net.http import NetBroadcast, NetFetch, NetReceive, NetSend, NetCallback
elif net_mode == "eth":
    from gfl.core.net.eth import NetBroadcast, NetFetch, NetReceive, NetSend, NetCallback
elif net_mode is None:
    from gfl.core.net.abstract import NetBroadcast, NetFetch, NetReceive, NetSend, NetCallback
    warnings.warn("net mode has not set.")
else:
    raise ValueError("unknown net mode.")
