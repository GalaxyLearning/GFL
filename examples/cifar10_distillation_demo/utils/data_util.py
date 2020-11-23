import os
import torch
from random import shuffle
from torchvision.datasets import utils, MNIST, CIFAR10, CIFAR100
from torchvision import transforms
from torch.utils.data import Subset, DataLoader
from PIL import Image

class Data(object):

    def __init__(self, dataset, split):
        # self.trainset, self.testset = None, None
        self.trainset, self.testset = Dataset(dataset)
        self.splited_trainset = data_split(self.trainset, split)
        # self.train_load = [DataLoader(splited_trainset[i], batch_size=batch_size, shuffle=True, num_workers=4)
        #                    for i in range(args.node_num)]
        # self.test_load = DataLoader(
        #     testset, batch_size=batch_size, shuffle=False, num_workers=4)
        if not os.path.exists("../data"):
            os.mkdir("../data")
        save_train_dataset(self.splited_trainset)
        save_test_dataset(self.testset)

def save_train_dataset(splited_trainset):
    for i in range(len(splited_trainset)):
        train_dataset_path = os.path.join("../data/train_dataset_{}".format(i))
        torch.save(splited_trainset[i], train_dataset_path)


def save_test_dataset(testset):
    test_dataset_path = os.path.join("../data/test_dataset")
    torch.save(testset, test_dataset_path)


def data_split(dataset, split):
    split_num = [int(len(dataset) / split) for _ in range(split)]
    split_cum = torch.tensor(list(split_num)).cumsum(dim=0).tolist()
    idx_dataset = list(range(len(dataset.targets)))
    # split by class
    # idx_dataset = sorted(range(len(dataset.targets)), key=lambda k: dataset.targets[k])
    # split by random
    splited_data = [idx_dataset[off - l:off]
                    for off, l in zip(split_cum, split_num)]
    splited = [Subset(dataset, splited_data[i]) for i in range(split)]
    return splited

class FEMNIST(MNIST):
    """
    This dataset is derived from the Leaf repository
    (https://github.com/TalwalkarLab/leaf) pre-processing of the Extended MNIST
    dataset, grouping examples by writer. Details about Leaf were published in
    "LEAF: A Benchmark for Federated Settings" https://arxiv.org/abs/1812.01097.
    """
    resources = [
        ('https://raw.githubusercontent.com/tao-shen/FEMNIST_pytorch/master/femnist.tar.gz',
         '59c65cec646fc57fe92d27d83afdf0ed')]

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        super(MNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')
        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        self.data, self.targets, self.users_index = torch.load(os.path.join(self.processed_folder, data_file))

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='F')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def download(self):
        """Download the FEMNIST data if it doesn't exist in processed_folder already."""
        import shutil

        if self._check_exists():
            return

        utils.makedir_exist_ok(self.raw_folder)
        utils.makedir_exist_ok(self.processed_folder)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            utils.download_and_extract_archive(url, download_root=self.raw_folder, filename=filename, md5=md5)

        # process and save as torch files
        print('Processing...')
        shutil.move(os.path.join(self.raw_folder, self.training_file), self.processed_folder)
        shutil.move(os.path.join(self.raw_folder, self.test_file), self.processed_folder)


def Dataset(dataset):
    trainset, testset = None, None
    if dataset == 'cifar10' or 'cifar100':
        tra_trans = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        val_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        if dataset == 'cifar10':
            trainset = CIFAR10(root="~/data", train=True,
                               download=True, transform=tra_trans)
            testset = CIFAR10(root="~/data", train=False,
                              download=True, transform=val_trans)
        if dataset == 'cifar100':
            trainset = CIFAR100(root="~/data", train=True,
                                download=True, transform=tra_trans)
            testset = CIFAR100(root="~/data", train=False,
                               download=True, transform=val_trans)
    if dataset == 'femnist' or 'mnist':
        tra_trans = transforms.Compose([
            transforms.Pad(2, padding_mode='edge'),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        val_trans = transforms.Compose([
            transforms.Pad(2, padding_mode='edge'),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        if dataset == 'femnist':
            trainset = FEMNIST(root='~/data', train=True, transform=tra_trans, digitsonly=False)
            testset = FEMNIST(root='~/data', train=False, transform=val_trans, digitsonly=False)
        if dataset == 'mnist':
            trainset = MNIST(root='~/data', train=True, transform=tra_trans)
            testset = MNIST(root='~/data', train=False, transform=val_trans)
    return trainset, testset