import os

from torchvision import datasets, transforms
from dataset_utils import *


class MnistDataset(datasets.MNIST):

    def __init__(self, root):
        datasets.MNIST.resources = [
            ('https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz',
             'f68b3c2dcbeaaa9fbdd348bbdeb94873'),
            ('https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz',
             'd53e105ee54ea40749a09fcbcd1e9432'),
            ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz',
             '9fb629c4189551a2d022fa330f9573f3'),
            ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz',
             'ec29112dd5afa0611ce80d1b7f02629c')
        ]
        super(MnistDataset, self).__init__(root, download=True, train=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.13066062,), (0.30810776,))
        ]))

abs_path = os.path.abspath("dataset")
mnist_dataset = MnistDataset(abs_path)

if __name__ == '__main__':
    # print(partition_iid(mnist_dataset, 10))
    # partition_noniid_(mnist_dataset, n_clients=20)
    vertical_partition_img(mnist_dataset, n_clients=10, shuffle=True)
