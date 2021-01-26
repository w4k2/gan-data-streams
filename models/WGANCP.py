import torch
import torch.nn as nn


class Generator(torch.nn.Module):
    def __init__(self, latent_vector_length=100, feature_map_size=64, color_channels=3):
        super().__init__()
        # Filters [1024, 512, 256]
        # Input_dim = 100
        # Output_dim = C (number of channels)
        self._module = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(latent_vector_length, feature_map_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_map_size * 8),
            nn.ReLU(True),
            # state size. (feature_map_size*8) x 4 x 4
            nn.ConvTranspose2d(feature_map_size * 8, feature_map_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 4),
            nn.ReLU(True),
            # state size. (feature_map_size*4) x 8 x 8
            nn.ConvTranspose2d(feature_map_size * 4, feature_map_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 2),
            nn.ReLU(True),
            # state size. (feature_map_size*2) x 16 x 16
            nn.ConvTranspose2d(feature_map_size * 2, feature_map_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size),
            nn.ReLU(True),
            # state size. (feature_map_size) x 32 x 32
            nn.ConvTranspose2d(feature_map_size, color_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (color_channels) x 64 x 64)
        )

    def forward(self, data):
        return self._module(data)


class Discriminator(torch.nn.Module):
    def __init__(self, feature_map_size=64, color_channels=3):
        super().__init__()
        self._feature_map_size = feature_map_size
        # Filters [256, 512, 1024]
        # Input_dim = channels (Cx64x64)
        # Output_dim = 1
        self._module = nn.Sequential(
            nn.Conv2d(color_channels, feature_map_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (feature_map_size) x 32 x 32
            nn.Conv2d(feature_map_size, feature_map_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (feature_map_size*2) x 16 x 16
            nn.Conv2d(feature_map_size * 2, feature_map_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (feature_map_size*4) x 8 x 8
            nn.Conv2d(feature_map_size * 4, feature_map_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (feature_map_size*8) x 4 x 4
            nn.Conv2d(feature_map_size * 8, 1, 4, 1, 0, bias=False)
        )

    def forward(self, data):
        return self._module(data)

    def feature_extraction(self, data):
        # Use discriminator for feature extraction then flatten to vector
        return self._module(data).view(-1, self._feature_map_size * 8 * 4 * 4)
