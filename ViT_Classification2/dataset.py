from torchvision.transforms import Compose, Resize, ToTensor
from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader


class Dataset:

    def __init__(self, downloadLocation, image_dim) -> None:
        transform = Compose([Resize((image_dim[1], image_dim[2])), ToTensor()])
        self.train_set = MNIST(
            root=downloadLocation, train=True, download=True, transform=transform)
        self.test_set = MNIST(root=downloadLocation, train=False,
                              download=True, transform=transform)

        self.train_loader = DataLoader(
            self.train_set, shuffle=True, batch_size=128)
        self.test_loader = DataLoader(
            self.test_set, shuffle=False, batch_size=128)
