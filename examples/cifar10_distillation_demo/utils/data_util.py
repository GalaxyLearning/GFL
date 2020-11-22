import os
import torch
from random import shuffle
from torchvision.datasets import utils, MNIST, CIFAR10, CIFAR100
from torchvision import transforms
from torch.utils.data import Subset, DataLoader


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
                                download=False, transform=tra_trans)
            testset = CIFAR100(root="~/data", train=False,
                               download=False, transform=val_trans)
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
        # if args.dataset == 'femnist':
        #     trainset = FEMNIST(root='~/data', train=True, transform=tra_trans, digitsonly=False)
        #     testset = FEMNIST(root='~/data', train=False, transform=val_trans, digitsonly=False)
        if dataset == 'mnist':
            trainset = MNIST(root='~/data', train=True, transform=tra_trans)
            testset = MNIST(root='~/data', train=False, transform=val_trans)
    return trainset, testset