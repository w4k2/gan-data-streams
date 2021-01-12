import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.utils import shuffle
import numpy as np
import torch


class Cifar100Dataset:

    def load_dataset(self):

        dataset = datasets.CIFAR100(transform=transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
            download=True, root='./data/datasets/cifar100')

        return dataset

