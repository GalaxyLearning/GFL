from pathlib import PurePath

import ipfshttpclient

from gfl.conf import GflConf
from gfl.utils.path_utils import PathUtils


ipfs_addr = GflConf.get_property("ipfs.addr")
tmp_dir = GflConf.get_property("dir.tmp")


class Ipfs(object):

    @classmethod
    def put(cls, file_bytes: bytes):
        """
        Adds file bytes to IPFS and return hash of the added IPFS object.

        :param file_bytes: file bytes
        :return: hash of the added IPFS object
        """
        client = ipfshttpclient.connect(ipfs_addr)
        res = client.add_bytes(file_bytes)
        return res["Hash"]

    @classmethod
    def get(cls, ipfs_hash: str):
        """
        Downloads a file from IPFS and read from the file.

        :param ipfs_hash: hash of the IPFS object
        :return: content read from the downloaded file
        """
        client = ipfshttpclient.connect(ipfs_addr)
        client.get(ipfs_hash, target=GflConf.cache_dir)
        path = PathUtils.join(GflConf.cache_dir, ipfs_hash)
        with open(path, "rb") as f:
            ret = f.read()
        return ret
