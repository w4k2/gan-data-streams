from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as dset


class DataProvider:

    def get_celeba_dataloader(self, image_size=64, batch_size=128, num_workers=2):
        dataset = dset.ImageFolder(root='./data/celeba',
                                   transform=transforms.Compose([
                                       transforms.Resize(image_size),
                                       transforms.CenterCrop(image_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        return dataloader
