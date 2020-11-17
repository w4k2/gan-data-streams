from models import Generator, Discriminator
from torch import nn
import torch

class GANTrainer:

    def __init__(self, n_gpu = 0):

        device = torch.device("cuda:0" if (torch.cuda.is_available() and n_gpu > 0) else "cpu")

        self._generator = Generator(n_gpu=n_gpu).to(device)
        self._discriminator = Discriminator(n_gpu=n_gpu).to(device)

        if (device.type == 'cuda') and (n_gpu > 1):
            self._generator = nn.DataParallel(self._generator, list(range(n_gpu)))
            self._discriminator = nn.DataParallel(self._discriminator, list(range(n_gpu)))

        self._generator.apply(self.init_weights)
        self._discriminator.apply(self.init_weights)

    def init_weights(self, model):
        classname = model.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(model.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(model.weight.data, 1.0, 0.02)
            nn.init.constant_(model.bias.data, 0)


