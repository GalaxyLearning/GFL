import torch
from torchvision import datasets, transforms
from pfl.core.client import FLClient
from pfl.core.strategy import WorkModeStrategy, TrainStrategy, LossStrategy
from pfl.core.trainer_controller import TrainerController

CLIENT_ID = 0

if __name__ == "__main__":
    # CLIENT_ID = int(sys.argv[1])

    mnist_data = datasets.MNIST("./mnist_data", download=True, train=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.13066062,), (0.30810776,))
    ]))
    client = FLClient()
    pfl_models = client.get_remote_pfl_models()

    for pfl_model in pfl_models:
        optimizer = torch.optim.SGD(pfl_model.get_model().parameters(), lr=0.01, momentum=0.5)
        train_strategy = TrainStrategy(optimizer=optimizer, batch_size=32, loss_function=LossStrategy.NLL_LOSS)
        pfl_model.set_train_strategy(train_strategy)

    TrainerController(work_mode=WorkModeStrategy.WORKMODE_STANDALONE, models=pfl_models, data=mnist_data, client_id=CLIENT_ID,
                      curve=True, concurrent_num=3).start()
