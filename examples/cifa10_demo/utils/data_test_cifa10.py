import dill
import os
import torch
import torchvision

from torchvision import datasets, transforms



def save_split_dataset(dataset_name, train_datasets, test_datasets):

    print("正在保存切割数据集....")
    dataset_dir = os.path.join(os.path.abspath("../"), dataset_name)

    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)

    train_dataset_dir = os.path.join(dataset_dir, "train_dataset_dir")
    if not os.path.exists(train_dataset_dir):
        os.mkdir(train_dataset_dir)

    test_dataset_dir = os.path.join(dataset_dir, "test_dataset_dir")
    if not os.path.exists(test_dataset_dir):
        os.mkdir(test_dataset_dir)

    train_dataset_name = "train_dataset"
    for idx, dataset in enumerate(train_datasets):
        dataset_name = train_dataset_name+"_{}".format(idx)
        dataset_path = os.path.join(train_dataset_dir, dataset_name)
        if not os.path.exists(dataset_path):
            # with open(dataset_path, "wb") as f:
            #     dill.dump(dataset, f)
                torch.save(dataset, dataset_path)

    if test_datasets is not None:
        test_dataset_name = "test_dataset"
        for idx, dataset in enumerate(test_datasets):
            dataset_name = test_dataset_name + "_{}".format(idx)
            dataset_path = os.path.join(test_dataset_dir, dataset_name)
            if not os.path.exists(dataset_path):
                # with open(dataset_path, "wb") as f:
                #     dill.dump(dataset, f)
                torch.save(dataset, dataset_path)

    print("数据集保存完成")


def save_test_dataset(dataset_name, test_dataset):
    dataset_dir = os.path.join(os.path.abspath("../"), dataset_name)
    test_dataset_dir = os.path.join(dataset_dir, "test_dataset_dir")
    test_dataset_path = os.path.join(test_dataset_dir, "test_dataset")
    torch.save(test_dataset, test_dataset_path)

def data_split():

    cifa10_test_dataset = torchvision.datasets.CIFAR10("../CIFA10", download=True, train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))
    # mnist_data = datasets.MNIST("../mnist_data", download=True, train=True, transform=transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.13066062,), (0.30810776,))
    # ]))

    # cifa10_len = len(cifa10_dataset)
    #
    # client_1_dataset, client_2_dataset, client_3_dataset = torch.utils.data.random_split(cifa10_dataset, [int(cifa10_len*0.3), int(cifa10_len*0.3), int(cifa10_len*0.4)])
    # train_datasets = [client_1_dataset, client_2_dataset, client_3_dataset]
    # save_split_dataset("cifa10", train_datasets, None)

    save_test_dataset("cifa10", cifa10_test_dataset)
    




if __name__ == "__main__":
    data_split()