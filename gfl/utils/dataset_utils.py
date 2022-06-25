from typing import Union, List
import matplotlib.pyplot as plt
import numpy as np
import torch


def partition_iid(dataset, n_clients):
    f"""
    
    Parameters
    ----------
    dataset: 需要进行划分的数据
    n_clients: 数据划分的份数

    Returns
    -------
    
    """
    n_samples = len(dataset)
    if n_samples < n_clients:
        raise ValueError(
            """Number of samples in dataset is less than n_clients"""
        )
    num_items = int(n_samples / n_clients)
    dict_clients, all_idxs = {}, np.arange(n_samples)
    # if replace:
    #     for i in range(n_clients):
    #         dict_clients[i] = set(np.random.choice(all_idxs, num_items, replace=True))
    #         all_idxs = list(set(all_idxs) - dict_clients[i])
    # else:
    all_idxs = np.random.permutation(all_idxs)
    batch_idxs = np.array_split(all_idxs, n_clients)
    for client_idx, batch_idx in enumerate(batch_idxs):
        dict_clients[client_idx] = batch_idx
    return dict_clients, all_idxs


def partition_noniid(dataset, n_clients):
    dict_clients = {i: np.array([]) for i in range(n_clients)}
    n_samples = len(dataset)
    dict_clients = {i: np.array([], dtype='int64') for i in range(n_clients)}
    labels = dataset.train_labels.numpy()
    indices = np.argsort(labels, axis=0).reshape((labels.shape[0]))
    labels = labels[indices]
    classes, start_indices = np.unique(labels, return_index=True)
    n_classes = len(classes)
    class_indices = np.random.shuffle(np.arange(n_classes))
    partitioned_classes = np.array_split(class_indices, n_clients)
    for i in range(n_clients):
        for rnd_class in partitioned_classes[i]:
            start_idx = start_indices[rnd_class]
            end_idx = start_indices[rnd_class + 1] if rnd_class != n_samples else n_samples
            dict_clients[i] = np.concatenate(
                (dict_clients[i], indices[start_idx: end_idx]), axis=0)
    return dict_clients, indices


def partition_noniid_(dataset,
                      dirichlet_dist: np.ndarray = None,
                      n_clients: int = 100,
                      concentration: Union[float, np.ndarray, List[float]] = 0.5,
                      accept_imbalanced: bool = False,
                      ):
    dict_clients = {i: np.array([]) for i in range(n_clients)}
    n_samples = len(dataset)
    dict_clients = {i: np.array([], dtype='int64') for i in range(n_clients)}
    labels = dataset.train_labels.numpy()
    indices = np.argsort(labels, axis=0).reshape((labels.shape[0]))
    labels = labels[indices]
    classes, start_indices = np.unique(labels, return_index=True)
    n_classes = len(classes)
    if (n_samples % n_clients) and (not accept_imbalanced):
        raise ValueError(
            """Total number of samples must be a multiple of `num_partitions`.
               If imbalanced classes are allowed, set
               `accept_imbalanced=True`."""
        )
    num_samples = n_clients * [0]
    for j in range(n_samples):
        num_samples[j % n_clients] += 1

    # Make sure that concentration is np.array and
    # check if concentration is appropriate
    concentration = np.asarray(concentration)

    if float("inf") in concentration:
        return partition_iid(dataset, n_clients)

    if concentration.size == 1:
        concentration = np.repeat(concentration, classes.size)
    elif concentration.size != classes.size:
        raise ValueError(
            f"The size of the provided concentration ({concentration.size}) ",
            f"must be either 1 or equals number of classes {classes.size})",
        )
    if dirichlet_dist is None:
        dirichlet_dist = np.random.default_rng().dirichlet(
            alpha=concentration, size=n_clients
        )
    if dirichlet_dist.size != 0:
        if dirichlet_dist.shape != (n_clients, classes.size):
            raise ValueError(
                f"""The shape of the provided dirichlet distribution
                     ({dirichlet_dist.shape}) must match the provided number
                      of partitions and classes ({n_clients},{classes.size})"""
            )
    plot_dist(dirichlet_dist)
    percentage = dirichlet_dist / dirichlet_dist.sum(axis=0)
    cumsum_percentage = np.cumsum(percentage, axis=0)
    n_class_samples = [0 for i in range(n_classes)]
    for class_idx in range(n_classes):
        if class_idx == n_classes - 1:
            n_class_samples[class_idx] = n_samples - start_indices[class_idx]
        else:
            n_class_samples[class_idx] = start_indices[class_idx+1] - start_indices[class_idx]
    pre_end_indices = start_indices
    for client_idx in range(n_clients):
        client_sample_indices = np.array([], dtype='int64')
        end_percentage = cumsum_percentage[client_idx]
        for class_idx in range(n_classes):
            start_idx = pre_end_indices[class_idx]
            end_idx = start_idx + int(end_percentage[class_idx] * n_class_samples[class_idx])
            pre_end_indices[class_idx] = end_idx
            client_sample_indices = np.concatenate((client_sample_indices, indices[start_idx:end_idx]), axis=0)
        dict_clients[client_idx] = client_sample_indices
    return dict_clients


def plot_dist(dirichlet_dist):
    dirichlet_dist_ = dirichlet_dist.transpose()
    cumsum_dist = np.cumsum(dirichlet_dist_, axis=0)
    n_classes, n_clients = dirichlet_dist_.shape
    for class_idx in range(n_classes):
        if class_idx == 0:
            plt.barh(range(n_clients), dirichlet_dist_[0])
        else:
            plt.barh(range(n_clients), dirichlet_dist_[class_idx], left=cumsum_dist[class_idx-1])
    plt.title("Distribution of Samples")
    plt.show()


def get_split_indices(indices, group_num=2, shuffle=True):
    if shuffle:
        np.random.shuffle(indices)
    split_indices = np.array_split(indices, group_num)
    return split_indices


def vertical_partition_img(dataset, n_clients=2, shuffle=False):
    idx = 0
    split_indices = None
    dic_single_datasets = {}
    for client_idx in range(n_clients):
        """
        Each value is a list of three elements, to accomodate, in order: 
        - data examples (as tensors)
        - label
        - index 
        """
        dic_single_datasets[client_idx] = []
    label_list = []
    index_list = []
    for tensor, label in dataset:
        if split_indices is None:
            height = tensor.shape[-1]
            height_indices = np.arange(height)
            split_indices = get_split_indices(height_indices, group_num=n_clients, shuffle=shuffle)
        for client_idx in range(n_clients):
            indices = torch.tensor(split_indices[client_idx])
            dic_single_datasets[client_idx].append(tensor[:, :, indices])
        label_list.append(torch.Tensor([label]))
        index_list.append(torch.Tensor([idx]))
        idx += 1

    return dic_single_datasets, label_list, index_list