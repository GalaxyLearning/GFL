# Copyright (c) 2019 GalaxyLearning Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# federate strategies
import torch
from enum import Enum
from gfl.exceptions.fl_expection import GFLException


class WorkModeStrategy(Enum):
    WORKMODE_STANDALONE = "standalone"
    WORKMODE_CLUSTER = "cluster"


class FederateStrategy(Enum):
    FED_AVG = "fed_avg"
    FED_DISTILLATION = "fed_distillation"

class LossStrategy(Enum):
    L1_LOSS = "L1loss"
    MSE_LOSS = "MSELoss"
    CROSSENTROPY_LOSS = "CrossEntropyLoss"
    NLL_LOSS = "NLLLoss"
    POISSIONNLL_LOSS = "PoissonNLLLoss"
    KLDIV_LOSS = "KLDivLoss"
    BCE_LOSS = "BCELoss"
    BCEWITHLOGITS_Loss = "BCEWithLogitsLoss"
    MARGINRANKING_Loss = "MarginRankingloss"

class SchedulerStrategy(Enum):
    CYCLICLR = "CyclicLR"
    COSINEANNEALINGLR = "CosineAnnealingLR"
    EXPONENTIALLR = "ExponentialLR"
    LAMBDALR = "LambdaLR"
    MULTISTEPLR = "ReduceLROnPlateau"
    STEPLR = "StepLR"

class OptimizerStrategy(Enum):
    OPTIM_SGD = "SGD"
    OPTIM_ADAM = "Adam"




class Strategy(object):
    def __init__(self):
        pass



class TrainStrategy(Strategy):

    def __init__(self, optimizer=None, scheduler=None, loss_function=LossStrategy.NLL_LOSS,
                 batch_size=0):
        super(TrainStrategy, self).__init__()
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.scheduler = scheduler

    def get_loss_functions(self):
        return LossStrategy.__members__.items()

    def get_fed_strategies(self):
        return FederateStrategy.__members__.items()

    def get_optim_strategies(self):
        return OptimizerStrategy.__members__.items()

    def get_scheduler_strategies(self):
        return SchedulerStrategy.__members__.items()

    def set_optimizer(self, optimizer):
        optim_strategies = self.get_optim_strategies()
        if optimizer in optim_strategies:
            self.optimizer = optimizer
        else:
            raise GFLException("optimizer strategy not found")

    def get_optimizer(self):
        return self.optimizer

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def get_scheduler(self):
        return self.scheduler


    def set_loss_function(self, loss_function):
        loss_functions = self.get_loss_functions()
        if loss_function in loss_functions:
            self.loss_function = loss_function.value
        else:
            raise GFLException("loss strategy not found")

    def get_loss_function(self):
        return self.loss_function

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def get_batch_size(self):
        return self.batch_size



class TestStrategy(Strategy):

    def __init__(self):
        super(TestStrategy, self).__init__()
