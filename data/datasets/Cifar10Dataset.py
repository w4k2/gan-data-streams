import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.utils import shuffle
import numpy as np


class Cifar10Dataset:

    def load_dataset(self):

        dataset = datasets.CIFAR10(transform=transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
            download=True, root='./data/datasets/cifar10')

        dataset.targets = np.asarray(dataset.targets)
        mask = dataset.targets > 4

        x1 = dataset.data[~mask]
        y1 = dataset.targets[~mask]
        x2 = dataset.data[mask]
        y2 = dataset.targets[mask]

        x1, y1 = shuffle(x1, y1)
        x2, y2 = shuffle(x2, y2)

        x1_split = np.split(x1, 2)
        y1_split = np.split(y1, 2)
        x2_split = np.split(x2, 2)
        y2_split = np.split(y2, 2)

        x = np.concatenate((x1_split[0], x2_split[0], x1_split[1], x2_split[1]))
        y = np.concatenate((y1_split[0], y2_split[0], y1_split[1], y2_split[1]))

        dataset.data = x
        dataset.targets = y

        return dataset

