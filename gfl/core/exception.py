class GFLException(Exception):

    def __init__(self, desc):
        super(GFLException, self).__init__()
        self.desc = desc

    def __str__(self):
        print(self.desc)