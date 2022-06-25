__all__ = [
    "FLDataset"
]


class FLDataset(object):

    def __init__(self, root):
        super(FLDataset, self).__init__()
        self.root = root
