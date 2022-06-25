import networkx as nx
import numpy as np

from gfl.topology.base_topology_manager import BaseTopologyManager


class DeCentralizedTopologyManager(BaseTopologyManager):
    """
    去中心化的拓扑结构（规则）。

    Arguments:
        n (int): number of nodes in the topology.
        neighbor_num (int): number of neighbors for each manager
    """

    def __init__(self, n, neighbor_num=2):
        self.n = n
        self.neighbor_num = neighbor_num
        self.topology = []
        # 需要操作这个映射关系的函数,index->manager
        self.map = {}
        # index_num保存着当前已经分配映射关系的节点数
        self.index_num = 0

    def add_node_into_topology(self, node, index=-1):
        # 将index->node的映射存入map
        # 未指定节点的编号
        if index == -1:
            self.map[self.index_num] = node
            self.index_num += 1
        else:
            self.map[index] = node

    def get_index_by_node_address(self, node_address):
        for index, node in self.map.items():
            if node.address == node_address:
                return index
        return -1

    def generate_topology(self):
        # first generate a ring topology
        topology_ring = np.array(nx.to_numpy_matrix(nx.watts_strogatz_graph(self.n, 2, 0)), dtype=np.float32)

        # randomly add some links for each manager (symmetric)
        k = int(self.neighbor_num)
        topology_random_link = np.array(nx.to_numpy_matrix(nx.watts_strogatz_graph(self.n, k, 0)), dtype=np.float32)

        # generate symmetric topology
        topology_symmetric = topology_ring.copy()
        for i in range(self.n):
            for j in range(self.n):
                if topology_symmetric[i][j] == 0 and topology_random_link[i][j] == 1:
                    topology_symmetric[i][j] = topology_random_link[i][j]
        np.fill_diagonal(topology_symmetric, 1)
        self.topology = topology_symmetric

    def get_in_neighbor_weights(self, node_index):
        if node_index >= self.n:
            return []
        in_neighbor_weights = []
        for row_idx in range(len(self.topology)):
            in_neighbor_weights.append(self.topology[row_idx][node_index])
        return in_neighbor_weights

    def get_out_neighbor_weights(self, node_index):
        if node_index >= self.n:
            return []
        return self.topology[node_index]

    def get_in_neighbor_idx_list(self, node_index):
        neighbor_in_idx_list = []
        neighbor_weights = self.get_in_neighbor_weights(node_index)
        for idx, neighbor_w in enumerate(neighbor_weights):
            if neighbor_w > 0 and node_index != idx:
                neighbor_in_idx_list.append(idx)
        return neighbor_in_idx_list

    def get_out_neighbor_idx_list(self, node_index):
        neighbor_out_idx_list = []
        neighbor_weights = self.get_out_neighbor_weights(node_index)
        for idx, neighbor_w in enumerate(neighbor_weights):
            if neighbor_w > 0 and node_index != idx:
                neighbor_out_idx_list.append(idx)
        return neighbor_out_idx_list

    def get_out_neighbor_node_list(self, node_index):
        neighbor_out_node_list = []
        neighbor_out_idx_list = self.get_out_neighbor_idx_list(node_index)
        for i in range(len(neighbor_out_idx_list)):
            neighbor_out_node_list.append(self.map[neighbor_out_idx_list[i]])
        return neighbor_out_node_list

    def get_in_neighbor_node_list(self, node_index):
        neighbor_in_node_list = []
        neighbor_in_idx_list = self.get_in_neighbor_idx_list(node_index)
        for i in range(len(neighbor_in_idx_list)):
            neighbor_in_node_list.append(self.map[neighbor_in_idx_list[i]])
        return neighbor_in_node_list
