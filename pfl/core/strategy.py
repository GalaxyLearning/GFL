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
from enum import Enum

import pfl.exceptions.fl_expection as exceptions


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
    MARGINRANKING_Loss = "MarginRankinpfloss"


class OptimizerStrategy(Enum):
    OPTIM_SGD = "SGD"
    OPTIM_ADAM = "Adam"




class Strategy(object):
    def __init__(self):
        pass



class TrainStrategy(Strategy):

    def __init__(self, optimizer=None, loss_function=LossStrategy.NLL_LOSS,
                 batch_size=0):
        super(TrainStrategy, self).__init__()
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.batch_size = batch_size

    def get_loss_functions(self):
        loss_functions = [LossStrategy.L1_LOSS, LossStrategy.MSE_LOSS, LossStrategy.CROSSENTROPY_LOSS,
                          LossStrategy.NLL_LOSS, LossStrategy.POISSIONNLL_LOSS,
                          LossStrategy.KLDIV_LOSS, LossStrategy.BCE_LOSS, LossStrategy.BCEWITHLOGITS_Loss,
                          LossStrategy.MARGINRANKING_Loss]
        return loss_functions

    def get_fed_strategies(self):
        fed_strategies = [FederateStrategy.FED_AVG, FederateStrategy.FED_DISTILLATION]
        return fed_strategies

    def get_optim_strategies(self):
        optim_strategies = [OptimizerStrategy.OPTIM_SGD, OptimizerStrategy.OPTIM_ADAM]
        return optim_strategies

    def set_optimizer(self, optimizer):
        optim_strategies = self.get_optim_strategies()
        if optimizer in optim_strategies:
            self.optimizer = optimizer.value
        else:
            raise exceptions.PFLException("optimizer strategy not found")

    def get_optimizer(self):
        return self.optimizer


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



class TestStrategy(Strategy):

    def __init__(self):
        super(TestStrategy, self).__init__()
