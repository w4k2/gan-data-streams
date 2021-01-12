from sklearn.utils import shuffle
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import torch


class CelebADataset:

    def load_dataset(self):

        dataset = datasets.CelebA(transform=transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.CenterCrop((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
            download=True, root='./data/datasets/celeba')

        targets = torch.zeros(dataset.attr.shape[0])
        for i, elem in enumerate(dataset.attr):
            if elem[20] == 1:
                targets[i] = 1
            else:
                targets[i] = 0

        dataset.attr = targets
        mask = dataset.attr == 1

        x1 = dataset.filename[~mask]
        y1 = dataset.attr[~mask]
        x2 = dataset.filename[mask]
        y2 = dataset.attr[mask]

        x1, y1 = shuffle(x1, y1)
        x2, y2 = shuffle(x2, y2)

        # x1_split = np.array_split(x1, 2)
        # y1_split = np.array_split(y1, 2)
        # x2_split = np.split(x2, 2)
        # y2_split = np.split(y2, 2)

        x = np.concatenate((x1, x2))
        y = np.concatenate((y1, y2))

        dataset.filename = x
        y = torch.from_numpy(y)
        y = torch.reshape(y, (y.shape[0], 1))
        dataset.attr = y

        return dataset
