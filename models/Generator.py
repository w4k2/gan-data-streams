from torch import nn


class Generator(nn.Module):
    def __init__(self, latent_vector_length=100, feature_map_size=64, color_channels=3, n_gpu=0):
        super(Generator, self).__init__()
        self.n_gpu = n_gpu
        self._model = nn.Sequential(
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
            # state size. (color_channels) x 64 x 64
        )

        self._layers = list(self._model._modules.values())
        self._n_layers = len(self._layers)
        self._layer_activations = None

    def forward(self, input):
        self._layer_activations = [input] + [None] * self._n_layers
        for l_idx in range(self._n_layers):
            self._layer_activations[l_idx+1] = self._layers[l_idx].forward(self._layer_activations[l_idx])
        return self._layer_activations[self._n_layers]
