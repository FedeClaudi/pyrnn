import torch.nn as nn
import torch

from ._utils import npify, torchify
from ._rnn import RNNBase


class RNN(RNNBase):
    r"""
    h_{t}= \tanh(W_{i h} x_{t}+b_{i h}+W_{h h} h_{(t-1)}+b_{h h})
    """

    def __init__(
        self,
        input_size=1,
        output_size=1,
        n_units=50,
        dale_ratio=None,
        autopses=True,
        connectivity=None,
    ):
        # Initialize base RNN class
        RNNBase.__init__(
            self,
            input_size=input_size,
            output_size=output_size,
            n_units=n_units,
            dale_ratio=dale_ratio,
            autopses=autopses,
            connectivity=connectivity,
        )

        # Define layers
        self.rnn_layer = nn.RNN(
            input_size=input_size,
            hidden_size=n_units,
            batch_first=True,
        )

        self.output_layer = nn.Linear(n_units, output_size)
        self.output_activation = nn.Tanh()

        # Freeze output layer's weights
        for p in self.output_layer.parameters():
            p.requires_grad = False

        # Build
        self.build()

    def get_recurrent_weights(self):
        return npify(self.rnn_layer.weight_hh_l0)

    def set_recurrent_weights(self, weights):
        self.rnn_layer.weight_hh_l0 = nn.Parameter(torchify(weights))

    def forward(self, x, h=None):
        if h is None:
            h = self._initialize_hidden(x)

        out, h = self.rnn_layer(x, h)
        out = self.output_activation(self.output_layer(out))
        return out, h


class CustomRNN(RNNBase):
    def __init__(
        self,
        input_size=1,
        output_size=1,
        n_units=50,
        dale_ratio=None,
        autopses=True,
        connectivity=None,
    ):
        # Initialize base RNN class
        RNNBase.__init__(
            self,
            input_size=input_size,
            output_size=output_size,
            n_units=n_units,
            dale_ratio=dale_ratio,
            autopses=autopses,
            connectivity=connectivity,
        )

        # Define layers
        self.w_in = nn.Linear(input_size, n_units, bias=False)
        self.w_rec = nn.Linear(n_units, n_units, bias=True)
        self.w_out = nn.Linear(n_units, output_size, bias=False)

        # Initialize weights
        # self.w_in.weight = nn.init.uniform_(self.w_in.weight, -0.1, 0.1)

        # freeze parameters
        for layer in (self.w_in, self.w_out):
            for p in layer.parameters():
                p.requires_grad = False

        self.activation = nn.Tanh()

        # Build
        self.build()

    def get_recurrent_weights(self):
        return npify(self.w_rec.weight)

    def set_recurrent_weights(self, weights):
        self.w_rec.weight = nn.Parameter(torchify(weights))

    def forward(self, x, h=None):
        if h is None:
            h = self._initialize_hidden(x)

        # Compute hidden
        nbatch, length, ninp = x.shape
        out = torch.zeros(length, nbatch, ninp)
        for t in range(length):
            h = self.activation(self.w_rec(h) + self.w_in(x[:, t, :]))

            out[t, :, :] = self.w_out(h)

        return out.permute(1, 0, 2), h
