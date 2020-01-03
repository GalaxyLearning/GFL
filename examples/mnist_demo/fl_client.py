from torchvision import datasets, transforms
from pfl.core.strategy import WorkModeStrategy
from pfl.core.trainer_controller import TrainerController

CLIENT_ID = 0

if __name__ == "__main__":
    # CLIENT_ID = int(sys.argv[1])

    mnist_data = datasets.MNIST("./mnist_data", download=True, train=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.13066062,), (0.30810776,))
    ]))

    TrainerController(work_mode=WorkModeStrategy.WORKMODE_STANDALONE, data=mnist_data, client_id=CLIENT_ID,
                      concurrent_num=3).start()
