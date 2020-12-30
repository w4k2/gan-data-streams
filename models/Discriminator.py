from torch import nn


class Discriminator(nn.Module):
    def __init__(self, feature_map_size=64, color_channels=3, n_gpu=0):
        super(Discriminator, self).__init__()
        self.n_gpu = n_gpu
        self._model = nn.Sequential(
            # input is (color_channels) x 64 x 64
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
            nn.Conv2d(feature_map_size * 8, 2, 4, 1, 0, bias=False),
            nn.LogSoftmax(dim=1)
        )

        self._layers = list(self._model._modules.values())
        self._n_layers = len(self._layers)
        self._layer_activations = None

    def forward(self, input):
        self._layer_activations = [input] + [None] * self._n_layers
        for l_idx in range(self._n_layers):
            self._layer_activations[l_idx+1] = self._layers[l_idx].forward(self._layer_activations[l_idx])
        return self._layer_activations[self._n_layers]
