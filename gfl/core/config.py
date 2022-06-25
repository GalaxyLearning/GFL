from dataclasses import dataclass
from typing import Dict, Any, List

from gfl.core.strategy import *
from gfl.utils import ModuleUtils
from gfl.utils.po_utils import PlainObject


@dataclass
class ObjectMeta:

    name: str
    is_instance: bool = False
    is_builtin: bool = False
    args: Dict[str, Any] = None


class ConfigObject(PlainObject):

    name: str
    is_instance: bool = False
    is_builtin: bool = False
    args: Dict[str, Any]

    def __init__(self, **kwargs):
        super(ConfigObject, self).__init__(**kwargs)

    @classmethod
    def new_object(cls, *, module=None, obj=None, strategy=None, is_instance=None, **kwargs):
        if strategy is not None:
            name = strategy.value
            if is_instance is None:
                is_instance = False
            is_builtin = True
        else:
            name = ModuleUtils.get_name(module, obj)
            if name is None:
                raise ValueError("")
            if is_instance is None:
                is_instance = not (type(obj) == type)
            is_builtin = False
        args = kwargs.copy()
        return ConfigObject(name=name, is_instance=is_instance, is_builtin=is_builtin, args=args)


class Config(PlainObject):

    args: Dict[str, Any] = {}

    def __init__(self, *, module=None, **kwargs):
        super(Config, self).__init__(**kwargs)
        self.module = module

    def with_args(self, **kwargs):
        self.args = kwargs.copy()
        return self

    def _set_config_object(self, obj, **kwargs):
        if isinstance(obj, Enum):
            return ConfigObject.new_object(strategy=obj, **kwargs)
        else:
            return ConfigObject.new_object(module=self.module, obj=obj, **kwargs)

    def _get_config_object(self, obj: ConfigObject, strategy_type, *args, **kwargs):
        if obj is None:
            return None
        if obj.is_instance:
            if obj.is_builtin:
                strategy: StrategyAdapter = strategy_type(obj.name)
                return strategy.get_type()
            else:
                if self.module is None:
                    raise ValueError("")
                return getattr(self.module, obj.name)
        else:
            if obj.is_builtin:
                strategy: StrategyAdapter = strategy_type(obj.name)
                clazz = strategy.get_type()
            else:
                if self.module is None:
                    raise ValueError("")
                clazz = getattr(self.module, obj.name)
            kwargs_copy = kwargs.copy()
            for k, v in obj.args.items():
                kwargs_copy[k] = v
            return clazz(*args, **kwargs_copy)


class JobConfig(Config):

    trainer: ConfigObject
    aggregator: ConfigObject

    def with_trainer(self, trainer, **kwargs):
        self.trainer = self._set_config_object(trainer, **kwargs)
        return self

    def with_aggregator(self, aggregator, **kwargs):
        self.aggregator = self._set_config_object(aggregator, **kwargs)
        return self

    def get_trainer(self):
        return self._get_config_object(self.trainer, TrainerStrategy)

    def get_aggregator(self):
        return self._get_config_object(self.aggregator, AggregatorStrategy)


class TrainConfig(Config):
    epoch: int = 10
    batch_size: int = 32
    model: ConfigObject
    optimizer: ConfigObject
    lr_scheduler: ConfigObject = None
    loss: ConfigObject

    def with_epoch(self, epoch=10):
        self.epoch = epoch
        return self

    def with_batch_size(self, batch_size=32):
        self.batch_size = batch_size
        return self

    def with_model(self, model, **kwargs):
        self.model = self._set_config_object(model, **kwargs)
        return self

    def with_optimizer(self, optimizer, **kwargs):
        self.optimizer = self._set_config_object(optimizer, **kwargs)
        return self

    def with_lr_scheduler(self, scheduler, **kwargs):
        self.lr_scheduler = self._set_config_object(scheduler, **kwargs)
        return self

    def with_loss(self, loss, **kwargs):
        self.loss = self._set_config_object(loss, **kwargs)
        return self

    def get_epoch(self):
        return self.epoch

    def get_batch_size(self):
        return self.batch_size

    def get_model(self):
        return self._get_config_object(self.model, None)

    def get_optimizer(self, model):
        return self._get_config_object(self.optimizer, OptimizerStrategy, model.parameters())

    def get_lr_scheduler(self, optimizer):
        return self._get_config_object(self.lr_scheduler, LRSchedulerStrategy, optimizer)

    def get_loss(self):
        return self._get_config_object(self.loss, LossStrategy)


class AggregateConfig(Config):
    # aggregation round
    round: int = 10
    clients_per_round: int = 2
    do_validation: bool = False
    batch_size: int = 32
    loss: ConfigObject

    def with_round(self, round_):
        self.round = round_
        return self

    def with_clients_per_round(self, clients_per_round):
        self.clients_per_round = clients_per_round
        return self

    def with_batch_size(self, batch_size=32):
        self.batch_size = batch_size
        return self

    def with_loss(self, loss, **kwargs):
        self.loss = self._set_config_object(loss, **kwargs)
        return self

    def get_round(self):
        return self.round

    def get_batch_size(self):
        return self.batch_size

    def get_loss(self):
        return self._get_config_object(self.loss, LossStrategy)


class DatasetConfig(Config):

    dataset: ConfigObject
    val_dataset: ConfigObject = None
    val_rate: float = 0.2

    def with_dataset(self, dataset, **kwargs):
        self.dataset = self._set_config_object(dataset, **kwargs)
        return self

    def with_val_dataset(self, val_dataset, **kwargs):
        self.val_dataset = self._set_config_object(val_dataset, **kwargs)
        return self

    def with_val_rate(self, val_rating):
        self.val_rate = val_rating
        return self

    def get_dataset(self):
        return self._get_config_object(self.dataset, None)

    def get_val_dataset(self):
        return self._get_config_object(self.val_dataset, None)

    def get_val_rate(self):
        return self.val_rate


class TopologyConfig(Config):
    train_node_num: int = 0
    server_nodes: List[str] = []
    client_nodes: List[str] = []
    topology: List[List[int]] = [[]]
    index2node: List[str] = []
    isCentralized: bool = True

    def with_train_node_num(self, train_node_num):
        self.train_node_num = train_node_num
        return self

    def with_server_nodes(self, server_nodes):
        self.server_nodes = server_nodes
        return self

    def with_client_nodes(self, client_nodes):
        self.client_nodes = client_nodes
        return self

    def with_topology(self, topology):
        self.topology = topology
        return self

    def with_index2node(self, index2node):
        self.index2node = index2node
        return self

    def with_isCentralized(self, isCentralized):
        self.isCentralized = isCentralized
        return self

    def get_train_node_num(self):
        return self.train_node_num

    def get_server_nodes(self):
        return self.server_nodes

    def get_client_nodes(self):
        return self.client_nodes

    def get_topology(self):
        return self.topology

    def get_index2node(self):
        return self.index2node

    def get_isCentralized(self):
        return self.isCentralized
