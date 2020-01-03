from torchvision import datasets, transforms
from pfl.core.strategy import WorkModeStrategy
from pfl.core.trainer_controller import TrainerController

SERVER_URL = "http://127.0.0.1:9763"
CLIENT_IP = "127.0.0.1"
CLIENT_PORT = 8081
CLIENT_ID = 0



if __name__ == "__main__":
    mnist_data = datasets.MNIST("./mnist_data", download=True, train=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.13066062,), (0.30810776,))
    ]))

    TrainerController(work_mode=WorkModeStrategy.WORKMODE_CLUSTER, data=mnist_data, client_id=CLIENT_ID,
                      client_ip=CLIENT_IP, client_port=CLIENT_PORT,
                      server_url=SERVER_URL, concurrent_num=3).start()
