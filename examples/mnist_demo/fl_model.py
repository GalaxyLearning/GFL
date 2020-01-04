from torch import nn
import torch.nn.functional as F
import pfl.core.strategy as strategy
from pfl.core.job_manager import JobManager


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)


if __name__ == "__main__":
    train_code_strategy = strategy.TrainStrategyFatorcy(optimizer=strategy.RunTimeStrategy.OPTIM_SGD,
                                                        learning_rate=0.01,
                                                        loss_function=strategy.RunTimeStrategy.NLL_LOSS, batch_size=32,
                                                        epoch=3)

    model = Net()

    job_manager = JobManager()
    job = job_manager.generate_job(work_mode=strategy.WorkModeStrategy.WORKMODE_STANDALONE,
                                   train_strategy=train_code_strategy,
                                   fed_strategy=strategy.FederateStrategy.FED_DISTILLATION, model=Net, distillation_alpha=0.5)
    job_manager.submit_job(job, model)
