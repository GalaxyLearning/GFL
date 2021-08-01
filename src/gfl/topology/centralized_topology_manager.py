import numpy as np
from gfl.core.config import TopologyConfig
from gfl.topology.base_topology_manager import BaseTopologyManager


class CentralizedTopologyManager(BaseTopologyManager):
    """
    中心化的拓扑结构。
    """

    def __init__(self, topology_config: TopologyConfig):
        # 节点总数，中心化的场景下聚合节点只有1个
        self.n = topology_config.get_train_node_num() + 1
        self.train_node_num = topology_config.get_train_node_num()
        # 保存该job的server_address
        self.server_address_list = topology_config.get_server_nodes()
        self.client_address_list = topology_config.get_client_nodes()
        self.topology = topology_config.get_topology()
        # 用来保存index->node_address的映射关系
        # 默认index为0的节点为聚合节点即server
        # 默认训练节点即client的index从1,2...按顺序递增
        self.index2node = topology_config.get_index2node()

    def add_server(self, server_node, add_into_topology: bool):
        # 此方法仅在拓扑结构不存在server和client的情况下调用
        self.server_address_list.append(server_node.address)
        self.n += 1
        if add_into_topology is True:
            self._add_node_into_topology(server_node)

    def add_client(self, client_node, add_into_topology: bool):
        self.client_address_list.append(client_node.address)
        self.n += 1
        self.train_node_num += 1
        if add_into_topology is True:
            self._add_node_into_topology(client_node)

    def _add_node_into_topology(self, node):
        # 将index->node的映射存入map
        self.index2node.append(node.address)

    def get_index_by_node(self, node):
        for index in range(len(self.index2node)):
            if self.index2node[index] == node.address:
                return index
        return -1

    def generate_topology(self):
        # 目前仅支持在确定节点之后，调用此方法生成拓扑结构
        # 不支持在生成拓扑结构之后，再往拓扑结构中添加新的节点
        topology_graph = np.zeros([self.n, self.n], dtype=np.int32)
        np.fill_diagonal(topology_graph, 1)
        for i in range(self.n):
            topology_graph[0][i] = 1
        for i in range(self.n):
            topology_graph[i][0] = 1
        self.topology = topology_graph.tolist()

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

    def get_out_neighbor_node_address_list(self, node_index):
        neighbor_out_node_address_list = []
        neighbor_out_idx_list = self.get_out_neighbor_idx_list(node_index)
        for i in range(len(neighbor_out_idx_list)):
            neighbor_out_node_address_list.append(self.index2node[neighbor_out_idx_list[i]])
        return neighbor_out_node_address_list

    def get_in_neighbor_node_address_list(self, node_index):
        neighbor_in_node_address_list = []
        neighbor_in_idx_list = self.get_in_neighbor_idx_list(node_index)
        for i in range(len(neighbor_in_idx_list)):
            neighbor_in_node_address_list.append(self.index2node[neighbor_in_idx_list[i]])
        return neighbor_in_node_address_list
