from data.datasets import Cifar10Dataset, Cifar100Dataset, CelebADataset
from torch.utils.data import DataLoader, SubsetRandomSampler


class DataProvider:

    def get_cifar10_dataloader(self, batch_size=128):
        dataset = Cifar10Dataset()
        dataloader = DataLoader(dataset=dataset.load_dataset(), batch_size=batch_size, shuffle=False, drop_last=True)
        return dataloader

    def get_cifar100_dataloader(self, batch_size=128):
        dataset = Cifar100Dataset()
        dataloader = DataLoader(dataset=dataset.load_dataset(), batch_size=batch_size, shuffle=False, drop_last=True)
        return dataloader

    def get_celeba_dataloaders(self, batch_size=128):
        dataset = CelebADataset()
        dataloaders = [
            DataLoader(dataset=dataset.load_dataset(), batch_size=batch_size, shuffle=False, drop_last=True,
                       sampler=SubsetRandomSampler(indices=list(range(94509)))),
            DataLoader(dataset=dataset.load_dataset(), batch_size=batch_size, shuffle=False, drop_last=True,
                       sampler=SubsetRandomSampler(indices=list(range(94509, 162770)))),
        ]

        return dataloaders
