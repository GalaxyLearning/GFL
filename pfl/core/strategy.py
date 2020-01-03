# federate strategies
from enum import Enum
import pfl.exceptions.fl_expection as exceptions


class WorkModeStrategy(Enum):
    WORKMODE_STANDALONE = "standalone"
    WORKMODE_CLUSTER = "cluster"


class FederateStrategy(Enum):
    FED_AVG = "fed_avg"
    FED_DISTILLATION = "fed_distillation"


class RunTimeStrategy(Enum):
    L1_LOSS = "L1loss"
    MSE_LOSS = "MSELoss"
    CROSSENTROPY_LOSS = "CrossEntropyLoss"
    NLL_LOSS = "NLLLoss"
    POISSIONNLL_LOSS = "PoissonNLLLoss"
    KLDIV_LOSS = "KLDivLoss"
    BCE_LOSS = "BCELoss"
    BCEWITHLOGITS_Loss = "BCEWithLogitsLoss"
    MARGINRANKING_Loss = "MarginRankinpfloss"
    OPTIM_SGD = "SGD"
    OPTIM_ADAM = "Adam"


class StrategyFactory(object):
    def __init__(self):
        pass


class TrainStrategyFatorcy(StrategyFactory):

    def __init__(self, optimizer=RunTimeStrategy.OPTIM_SGD, learning_rate=0.01, loss_function=RunTimeStrategy.NLL_LOSS,
                 batch_size=0, epoch=10):
        super(StrategyFactory, self).__init__()
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.epoch = epoch

    def get_loss_functions(self):
        loss_functions = [RunTimeStrategy.L1_LOSS, RunTimeStrategy.MSE_LOSS, RunTimeStrategy.CROSSENTROPY_LOSS,
                          RunTimeStrategy.NLL_LOSS, RunTimeStrategy.POISSIONNLL_LOSS,
                          RunTimeStrategy.KLDIV_LOSS, RunTimeStrategy.BCE_LOSS, RunTimeStrategy.BCEWITHLOGITS_Loss,
                          RunTimeStrategy.MARGINRANKING_Loss]
        return loss_functions

    def get_fed_strategies(self):
        fed_strategies = [FederateStrategy.FED_AVG, FederateStrategy.FED_DISTILLATION]
        return fed_strategies

    def get_optim_strategies(self):
        optim_strategies = [RunTimeStrategy.OPTIM_SGD, RunTimeStrategy.OPTIM_ADAM]
        return optim_strategies

    def set_optimizer(self, optimizer):
        optim_strategies = self.get_optim_strategies()
        if optimizer in optim_strategies:
            self.optimizer = optimizer.value
        else:
            raise exceptions.PFLException("optimizer strategy not found")

    def get_optimizer(self):
        return self.optimizer

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def get_learning_rate(self):
        return self.learning_rate

    def set_loss_function(self, loss_function):
        loss_functions = self.get_loss_functions()
        if loss_function in loss_functions:
            self.loss_function = loss_function.value
        else:
            raise exceptions.PFLException("loss strategy not found")

    def get_loss_function(self):
        return self.loss_function

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def get_batch_size(self):
        return self.batch_size

    def set_epoch(self, epoch):
        self.epoch = epoch

    def get_epoch(self):
        return self.epoch




class TestStrategyFactory(StrategyFactory):

    def __init__(self):
        super(TestStrategyFactory, self).__init__()
