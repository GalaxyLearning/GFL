import os
import torch
from torchvision import datasets, transforms
from gfl.core.client import FLClient
from gfl.core.strategy import WorkModeStrategy, TrainStrategy, LossStrategy
from gfl.core.trainer_controller import TrainerController
import sys
sys.path.append(os.path.join(os.path.abspath("."), "utils"))
CLIENT_ID = 1

if __name__ == "__main__":
    # CLIENT_ID = int(sys.argv[1])

    dataset_path = os.path.join(os.path.abspath("."), "data",
                                "train_dataset_{}".format(CLIENT_ID))
    test_dataset_path = os.path.join(os.path.abspath("."), "data",
                                     "test_dataset")
    test_dataset = torch.load(test_dataset_path)
    dataset = torch.load(dataset_path)
    client = FLClient()
    gfl_models = client.get_remote_gfl_models()

    for gfl_model in gfl_models:
        optimizer = torch.optim.SGD(gfl_model.get_model().parameters(), lr=0.01, momentum=0.5, weight_decay=5e-4)
        train_strategy = TrainStrategy(optimizer=optimizer, batch_size=32, loss_function=LossStrategy.CE_LOSS)
        gfl_model.set_train_strategy(train_strategy)

    TrainerController(work_mode=WorkModeStrategy.WORKMODE_STANDALONE, models=gfl_models, data=dataset,
                      test_data=test_dataset, client_id=CLIENT_ID,
                      curve=False, local_epoch=3, concurrent_num=3).start()
