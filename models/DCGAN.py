from utils import utils
from torch import nn
import torch


class Generator(nn.Module):
    def __init__(self, latent_vector_length=100, feature_map_size=64, color_channels=3, device=torch.device("cpu")):
        super(Generator, self).__init__()
        self._device = device
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

        self._conv_indices = [0, 4, 8, 12, 16]

        self._layers = list(self._model._modules.values())
        self._n_layers = len(self._layers)
        self._layer_activations = None
        self._relevance_scores = None
        self._neuron_importance = None
        self._importance_level = 0.5

    def forward(self, data):
        self._layer_activations = [data] + [None] * self._n_layers
        for l_idx in range(self._n_layers):
            self._layer_activations[l_idx + 1] = self._layers[l_idx].forward(self._layer_activations[l_idx])

        return self._layer_activations[self._n_layers]

    def calculate_relevance(self, discriminator_pixel_relevance):

        self._relevance_scores = [None] * self._n_layers + [discriminator_pixel_relevance]

        for l_idx in range(1, self._n_layers)[::-1]:
            # Loop is not applicable to the pixel layer
            self._layer_activations[l_idx] = self._layer_activations[l_idx].data.requires_grad_(True)

            if isinstance(self._layers[l_idx], torch.nn.ConvTranspose2d):
                def rho(p):
                    return p

                def incr(z):
                    return z

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
                self._relevance_scores[l_idx] = self._relevance_scores[l_idx + 1]

        if self._neuron_importance is None:
            self._neuron_importance = [None] * (len(self._relevance_scores) - 1)

            for l_idx in range(1, self._n_layers)[::-1]:
                self._neuron_importance[l_idx] = torch.zeros(self._relevance_scores[l_idx].size(), device=self._device)

        for l_idx in range(1, self._n_layers)[::-1]:
            mask = self._relevance_scores[l_idx] > 0
            self._neuron_importance[l_idx][mask] += 1


class Discriminator(nn.Module):
    def __init__(self, feature_map_size=64, color_channels=3):
        super(Discriminator, self).__init__()
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
        self._relevance_scores = None
        self._neuron_importance = None

    def forward(self, input):
        self._layer_activations = [input] + [None] * self._n_layers
        for l_idx in range(self._n_layers):
            self._layer_activations[l_idx+1] = self._layers[l_idx].forward(self._layer_activations[l_idx])

        return self._layer_activations[-1]

    def get_pixel_layer_relevance(self):

        best_matching_neuron_idx = self._layer_activations[-1].argmax(1)
        last_layer_activations = torch.zeros(self._layer_activations[-1].shape)

        last_layer_activations[best_matching_neuron_idx] = 1.0

        self._relevance_scores = [None] * self._n_layers + [last_layer_activations.data]

        for l_idx in range(1, self._n_layers)[::-1]:
            # Loop is not applicable to the pixel layer, which requires different rule
            self._layer_activations[l_idx] = self._layer_activations[l_idx].data.requires_grad_(True)

            if isinstance(self._layers[l_idx], torch.nn.Conv2d):
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

        self._layer_activations[0] = self._layer_activations[0].data.requires_grad_(True)

        # Pixel layer calculations
        lower_bound = (self._layer_activations[0].data * 0 + 0).requires_grad_(True)
        higher_bound = (self._layer_activations[0].data * 0 + 1).requires_grad_(True)

        z_k = utils.newlayer(self._layers[0], lambda p: p).forward(self._layer_activations[0]) + 1e-9
        z_k -= utils.newlayer(self._layers[0], lambda p: p.clamp(min=0)).forward(lower_bound)
        z_k -= utils.newlayer(self._layers[0], lambda p: p.clamp(max=0)).forward(higher_bound)
        s_k = (self._relevance_scores[1] / z_k).data
        (z_k * s_k).sum().backward()
        c, cp, cm = self._layer_activations[0].grad, lower_bound.grad, higher_bound.grad
        self._relevance_scores[0] = (self._layer_activations[0] * c + lower_bound * cp + higher_bound * cm).data

        return self._relevance_scores[0]