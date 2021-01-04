from torch import nn


class RBDropout2D(nn.Module):
    def __init__(self):
        super(RBDropout2D, self).__init__()

        self._active = False
        self._mask = None

    def activate(self, activation):
        self._active = activation

    def set_mask(self, mask):
        self._mask = mask

    def forward(self, x):
        if not self._active:
            return x
        if self.training:
            return x * self._mask
        return x
