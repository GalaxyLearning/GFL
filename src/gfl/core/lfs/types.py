from collections import namedtuple


File = namedtuple("File", ["file", "ipfs_hash"], defaults=[None, None])
