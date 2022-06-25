from enum import Enum

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


class StrategyAdapter(object):

    def get_type(self):
        return self._torch_type()

    def _torch_type(self):
        raise NotImplementedError("")


class LossStrategy(StrategyAdapter, Enum):

    L1 = "L1Loss"
    NLL = "NLLLoss"
    POISSON_NLL = "PoissonNLLLoss"
    KL_DIV = "KLDivLoss"
    MSE = "MSELoss"
    BCE = "BCELoss"
    BCE_WITH_LOGITS = "BCEWithLogitsLoss"
    HINGE_EMBEDDING = "HingeEmbeddingLoss"
    MULTI_LABEL_MARGIN = "MultiLabelMarginLoss"
    SMOOTH_L1 = "SmoothL1Loss"
    SOFT_MARGIN = "SoftMarginLoss"
    CROSS_ENTROPY = "CrossEntropyLoss"
    MULTI_LABEL_SOFT_MARGIN = "MultiLabelSoftMarginLoss"
    COSINE_EMBEDDING = "CosineEmbeddingLoss"
    MARGIN_RANKING = "MarginRankingLoss"
    MULTI_MARGIN = "MultiMarginLoss"
    TRIPLE_MARGIN = "TripletMarginLoss"
    CTC = "CTCLoss"

    def _torch_type(self):
        return getattr(nn, self.value)


class OptimizerStrategy(StrategyAdapter, Enum):

    SGD = "SGD"
    ASGD = "ASGD"
    RPROP = "Rprop"
    ADAGRAD = "Adagrad"
    ADADELTA = "Adadelta"
    RMSprop = "RMSprop"
    ADAM = "Adam"
    ADAMW = "AdamW"
    ADAMAX = "Adamax"
    SPARSE_ADAM = "SparseAdam"
    LBFGS = "LBFGS"

    def _torch_type(self):
        return getattr(optim, self.value)


class LRSchedulerStrategy(StrategyAdapter, Enum):

    LAMBDA_LR = "LambdaLR"
    MULTIPLICATIVE_LR = "MultiplicativeLR"
    STEP_LR = "StepLR"
    MULTI_STEP_LR = "MultiStepLR"
    EXPONENTIAL_LR = "ExponentialLR"
    COSINE_ANNEALING_LR = "CosineAnnealingLR"
    ReduceLROnPlateau = "ReduceLROnPlateau"
    CYCLIC_LR = "CyclicLR"
    COSINE_ANNEALING_WARM_RESTARTS = "CosineAnnealingWarmRestarts"
    ONE_CYCLE_LR = "OneCycleLR"

    def _torch_type(self):
        return getattr(lr_scheduler, self.value)


class TrainerStrategy(StrategyAdapter, Enum):

    SUPERVISED = "SupervisedTrainer"

    def _torch_type(self):
        import gfl.core.fl.trainer as trainer
        return getattr(trainer, self.value)


class AggregatorStrategy(StrategyAdapter, Enum):

    FED_AVG = "FedAvgAggregator"

    def _torch_type(self):
        import gfl.core.fl.aggregator as aggregator
        return getattr(aggregator, self.value)
