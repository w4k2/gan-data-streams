from data.datasets import CelebADataset, CustomDataset
from torch.utils.data import DataLoader, SubsetRandomSampler


class DataProvider:

    def get_celeba_dataloaders(self, batch_size=128):
        dataset = CelebADataset().load_dataset()
        dataloaders = [
            DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=True,
                       sampler=SubsetRandomSampler(indices=list(range(143258)))),
            DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=True,
                       sampler=SubsetRandomSampler(indices=list(range(143258, 277006)))),
        ]

        return dataloaders

    def get_custom_dataloader(self, x, y, batch_size=128):
        dataset = CustomDataset(x, y)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        return dataloader
