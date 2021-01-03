from utils import utils
from torch import nn
import torch


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
        self._relevance_scores = None

    def forward(self, input):
        self._layer_activations = [input] + [None] * self._n_layers
        for l_idx in range(self._n_layers):
            self._layer_activations[l_idx+1] = self._layers[l_idx].forward(self._layer_activations[l_idx])
        return self._layer_activations[self._n_layers]

    def calculate_relevance(self, discriminator_pixel_relevance):

        self._relevance_scores = [None] * self._n_layers + [discriminator_pixel_relevance]

        for l_idx in range(1, self._n_layers)[::-1]:
            # Loop is not applicable to the pixel layer
            self._layer_activations[l_idx] = self._layer_activations[l_idx].data.requires_grad_(True)

            if isinstance(self._layers[l_idx], torch.nn.ConvTranspose2d):
                def rho(p): return p
                def incr(z): return z

                if l_idx <= 4:
                    def rho(p): return p + 0.25 * p.clamp(min=0)
                    def incr(z): return z + 1e-9
                if 5 <= l_idx <= 7:
                    def rho(p): return p
                    def incr(z): return z + 1e-9 + 0.25 * ((z ** 2).mean() ** .5).data
                if l_idx >= 8:
                    def rho(p): return p
                    def incr(z): return z + 1e-9

                # Four steps according to LRP paper
                #  G. Montavon, A. Binder, S. Lapuschkin, W. Samek, K.-R. Müller
                # Layer-wise Relevance Propagation: An Overview
                # in Explainable AI, Springer LNCS, vol. 11700, 2019
                z_k = incr(utils.newlayer(self._layers[l_idx], rho).forward(self._layer_activations[l_idx]))  # step 1
                s_k = (self._layer_activations[l_idx + 1] / z_k).data  # step 2
                (z_k * s_k).sum().backward()
                c_j = self._layer_activations[l_idx].grad  # step 3
                self._relevance_scores[l_idx] = (self._layer_activations[l_idx] * c_j).data  # step 4
            else:
                self._relevance_scores[l_idx] = self._relevance_scores[l_idx+1]

    def apply_dropout(self):
        pass
