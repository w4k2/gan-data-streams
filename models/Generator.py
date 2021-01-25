from models.RBDropout2D import RBDropout2D
from utils import utils
from torch import nn
import torch


class Generator(nn.Module):
    def __init__(self, latent_vector_length=100, feature_map_size=64, color_channels=3, n_gpu=0,
                 device=torch.device("cpu")):
        super(Generator, self).__init__()
        self.n_gpu = n_gpu
        self._device = device
        self._dropout_layers = [RBDropout2D() for _ in range(5)]
        self._model = nn.Sequential(
            # input is Z, going into a convolution
            self._dropout_layers[0],
            nn.ConvTranspose2d(latent_vector_length, feature_map_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_map_size * 8),
            nn.ReLU(True),
            # state size. (feature_map_size*8) x 4 x 4
            self._dropout_layers[1],
            nn.ConvTranspose2d(feature_map_size * 8, feature_map_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 4),
            nn.ReLU(True),
            # state size. (feature_map_size*4) x 8 x 8
            self._dropout_layers[2],
            nn.ConvTranspose2d(feature_map_size * 4, feature_map_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 2),
            nn.ReLU(True),
            # state size. (feature_map_size*2) x 16 x 16
            self._dropout_layers[3],
            nn.ConvTranspose2d(feature_map_size * 2, feature_map_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size),
            nn.ReLU(True),
            # state size. (feature_map_size) x 32 x 32
            # self._dropout_layers[4],
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

    def forward(self, input):
        self._layer_activations = [input] + [None] * self._n_layers
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

    def activate_dropout(self, activation):
        for dropout_layer in self._dropout_layers:
            dropout_layer.activate(activation)

    def apply_dropout(self):

        for conv_idx, dropout_layer in zip(self._conv_indices, self._dropout_layers):
            normalized_importance = self._neuron_importance[conv_idx + 1].clone()
            normalized_importance -= normalized_importance.min(1, keepdim=True)[0]
            normalized_importance /= normalized_importance.max(1, keepdim=True)[0]

            # max_rand = 0.05
            # min_rand = -0.05

            sample_mask = torch.zeros(normalized_importance.shape, dtype=torch.float, device=self._device)

            for batch_idx, batch in enumerate(normalized_importance):
                for channel_idx, channel in enumerate(batch):
                    mean = torch.mean(channel)
                    if mean <= self._importance_level:
                        sample_mask[batch_idx][channel_idx] = 1.0

            # random_mask = (max_rand - min_rand) * torch.rand(normalized_importance.shape, device=self._device) + min_rand
            # random_mask += self._importance_level
            # sample_mask[normalized_importance <= random_mask] = 1.0

            dropout_layer.set_mask(sample_mask)

        self._neuron_importance = None
