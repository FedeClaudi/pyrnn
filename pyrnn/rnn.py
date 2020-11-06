import torch.nn as nn

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
        free_output_weights=False,
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
            free_output_weights=free_output_weights,
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
        if free_output_weights:
            for p in self.output_layer.parameters():
                p.requires_grad = False

        # Build
        self.build()

    def get_recurrent_weights(self):
        return npify(self.rnn_layer.weight_hh_l0, flatten=False)

    def set_recurrent_weights(self, weights):
        self.rnn_layer.weight_hh_l0 = nn.Parameter(
            torchify(weights, flatten=False)
        )

    def forward(self, x, h=None):
        if h is None:
            h = self._initialize_hidden(x)

        out, h = self.rnn_layer(x, h)
        out = self.output_activation(self.output_layer(out))
        return out, h
