import torch
from gfl.core.client import FLClient
from torchvision import datasets, transforms
from gfl.core.trainer_controller import TrainerController
from gfl.core.strategy import WorkModeStrategy, TrainStrategy, LossStrategy

SERVER_URL = "http://127.0.0.1:9763"
CLIENT_IP = "127.0.0.1"
CLIENT_PORT = 8081
CLIENT_ID = 0



if __name__ == "__main__":
    mnist_data = datasets.MNIST("./mnist_data", download=True, train=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.13066062,), (0.30810776,))
    ]))

    client = FLClient()
    gfl_models = client.get_remote_gfl_models(SERVER_URL)

    for gfl_model in gfl_models:
        optimizer = torch.optim.SGD(gfl_model.get_model().parameters(), lr=0.01, momentum=0.5)
        train_strategy = TrainStrategy(optimizer=optimizer, batch_size=32, loss_function=LossStrategy.NLL_LOSS)
        gfl_model.set_train_strategy(train_strategy)

    TrainerController(work_mode=WorkModeStrategy.WORKMODE_CLUSTER, models=gfl_models, data=mnist_data, client_id=CLIENT_ID,
                      client_ip=CLIENT_IP, client_port=CLIENT_PORT,
                      server_url=SERVER_URL, curve=True, local_epoch=5, concurrent_num=3).start()
