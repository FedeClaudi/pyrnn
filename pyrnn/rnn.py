import torch.nn as nn
import torch

# from torch.nn.utils.rnn import pad_packed_sequence

from pyrnn._utils import npify, torchify
from pyrnn._rnn import RNNBase


class RNN(RNNBase):
    r"""
    Custom RNN class. Unlike TorchRNN it doesn't use
    pytorch's RNN class but builds similar functionality
    using linear layers.

    The hidden state is updated given:
        h_{t}= \tanh(W_{in} x_{t}+W_{rec} h_{(t-1)}+b_{rec})

    and the output is given by:
        o_{t} = W_{out} h_{t}

    """
    sigma = nn.Tanh()

    def __init__(
        self,
        input_size=1,
        output_size=1,
        n_units=50,
        dale_ratio=None,
        autopses=True,
        connectivity=None,
        input_connectivity=None,
        output_connectivity=None,
        w_in_bias=False,
        w_in_train=False,
        w_out_bias=False,
        w_out_train=False,
        on_gpu=False,
    ):
        """
        Implements custom RNN.

        Arguments:
            input_size (int): number of inputs
            output_size (int): number of outputs
            n_units (int): number of units in RNN layer
            dale_ratio (float): percentage of excitatory units.
                Use None to ignore
            autopses (bool): pass False to remove autopses from
                recurrent weights
            connectivity: np.array of shape n_units x n_units with connectivity
                constraints for the recurrent layer
            input_connectivity: np.array fo shape n_units x n_inputs with connectivity
                constraints for the input layer
            output_connectivity: np.array of shape n_units x n_outputs with connectivity
                constraints for output layer
            w_in_bias/w_out_bias (bool): if True in/out linear
                layers will have a bias
            w_in_train/w_out_train (bool): if True in/out linear
                layers will be trained
            on_gpu (bool): if true computation is carried out on a GPU

        """
        # Initialize base RNN class
        RNNBase.__init__(
            self,
            input_size=input_size,
            output_size=output_size,
            n_units=n_units,
            dale_ratio=dale_ratio,
            autopses=autopses,
            connectivity=connectivity,
            input_connectivity=input_connectivity,
            output_connectivity=output_connectivity,
            on_gpu=on_gpu,
        )
        self.w_in_bias = w_in_bias
        self.w_in_train = w_in_train
        self.w_out_bias = w_out_bias
        self.w_out_train = w_out_train

        # Define layers
        self.w_in = nn.Linear(input_size, n_units, bias=w_in_bias)
        self.w_rec = nn.Linear(n_units, n_units, bias=True)
        self.w_out = nn.Linear(n_units, output_size, bias=w_out_bias)

        # apply connectivity constraints to input and output layers
        for layer, constraint in zip(
            (self.w_in, self.w_out), (input_connectivity, output_connectivity)
        ):
            if constraint is not None:
                layer.weight = nn.Parameter(
                    layer.weight * torchify(constraint)
                )
                if layer.bias is not None:
                    layer.bias = nn.Parameter(
                        layer.bias * torchify(constraint)
                    )

        # freeze parameters for input/output layers
        for layer, trainable in zip(
            (self.w_out, self.w_out), (w_in_train, w_out_train)
        ):
            if not trainable:
                for p in layer.parameters():
                    p.requires_grad = False

        # Build
        self.build()

        self.params.update(
            dict(
                w_in_bias=self.w_in_bias,
                w_in_train=self.w_in_train,
                w_out_bias=self.w_out_bias,
                w_out_train=self.w_out_train,
            )
        )

    def get_recurrent_weights(self):
        """
        Get weights of recurrent layer
        """
        return npify(self.w_rec.weight)

    def set_recurrent_weights(self, weights):
        """
        Set weights of recurrent layer

        Arguments:
            weights (np.ndarray): (n_units * n_units) 2d array with recurrent weights
        """
        self.w_rec.weight = nn.Parameter(torchify(weights))

    def forward(self, x, h=None):
        """
        Specifies how the forward dynamics of the network are computed

        Arguments:
            x (np.ndarray): network's input
            h (torch tensor) tensor with initial hidden state.
                If None the state is initialized as 0s.

        Returns:
            out (np.ndarray): the network's output
            h (torch tensor): updated hidden stae
        """
        if h is None:
            h = self._initialize_hidden(x)

        if self.on_gpu:
            h = h.cuda(0)

        # Compute hidden
        nbatch, length, ninp = x.shape
        out = torch.zeros(length, nbatch, self.output_size)
        for t in range(length):
            h = self.sigma(self.w_rec(h) + self.w_in(x[:, t, :]))

            out[t, :, :] = self.w_out(h)

        return out.permute(1, 0, 2), h


class CTRNN(RNN):
    r"""
    Custom RNN class for continuous time RNN.

    The hidden state is updated given:
        \tau \dot{h}_{t}= W_{in} * x_{t} + W_{rec} \sigma(h_{t-1}) + b_{rec})

    and the output is given by:
        o_{t} = W_{out} h_{t}

    """
    sigma = nn.Tanh()

    def __init__(
        self,
        input_size=1,
        output_size=1,
        n_units=50,
        dale_ratio=None,
        autopses=True,
        connectivity=None,
        input_connectivity=None,
        output_connectivity=None,
        w_in_bias=False,
        w_in_train=False,
        w_out_bias=False,
        w_out_train=False,
        on_gpu=False,
        dt=5,
        tau=100,
    ):
        """
        Implements custom RNN.

        Arguments:
            input_size (int): number of inputs
            output_size (int): number of outputs
            n_units (int): number of units in RNN layer
            dale_ratio (float): percentage of excitatory units.
                Use None to ignore
            autopses (bool): pass False to remove autopses from
                recurrent weights
            connectivity: np.array of shape n_units x n_units with connectivity
                constraints for the recurrent layer
            input_connectivity: np.array fo shape n_units x n_inputs with connectivity
                constraints for the input layer
            output_connectivity: np.array of shape n_units x n_outputs with connectivity
                constraints for output layer
            w_in_bias/w_out_bias (bool): if True in/out linear
                layers will have a bias
            w_in_train/w_out_train (bool): if True in/out linear
                layers will be trained
            on_gpu (bool): if true computation is carried out on a GPU
            dt: int. Time interval between two consecutive samples
            tau: int. Time constant of the network's hidden dynamics

        """
        # Initialize base RNN class
        RNN.__init__(
            self,
            input_size=input_size,
            output_size=output_size,
            n_units=n_units,
            dale_ratio=dale_ratio,
            autopses=autopses,
            connectivity=connectivity,
            input_connectivity=input_connectivity,
            output_connectivity=output_connectivity,
            on_gpu=on_gpu,
        )

        self.tau, self.dt = tau, dt

        self.params.update(
            dict(
                tau=self.tau,
                dt=self.dt,
            )
        )

    def forward(self, x, h=None):
        """
        Specifies how the forward dynamics of the network are computed

        Arguments:
            x (np.ndarray): network's input
            h (torch tensor) tensor with initial hidden state.
                If None the state is initialized as 0s.

        Returns:
            out (np.ndarray): the network's output
            h (torch tensor): updated hidden stae
        """
        if h is None:
            h = self._initialize_hidden(x)

        if self.on_gpu:
            h = h.cuda(0)

        # Compute hidden
        nbatch, length, ninp = x.shape
        out = torch.zeros(length, nbatch, self.output_size)
        for t in range(length):
            hdot = (
                -h + self.sigma(self.w_rec(h) + self.w_in(x[:, t, :]))
            ) / self.tau
            h = h + self.dt * hdot

            out[t, :, :] = self.w_out(h)

        return out.permute(1, 0, 2), h


class TorchRNN(RNNBase):
    r"""
    Implements a basic RNN using the RNN class from pytorch and
    a dense linear layer for output.

    The RNN layer's hidden state is given by:
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
        input_connectivity=None,
        output_connectivity=None,
    ):
        """
        Implements a basic RNN using the RNN class from pytorch and
            a dense linear layer for output.

        Arguments:
            input_size (int): number of inputs
            output_size (int): number of outputs
            n_units (int): number of units in RNN layer
            dale_ratio (float): percentage of excitatory units.
                Use None to ignore
            autopses (bool): pass False to remove autopses from
                recurrent weights
            connectivity: np.array of shape n_units x n_units with connectivity
                constraints for the recurrent layer
            input_connectivity: np.array fo shape n_units x n_inputs with connectivity
                constraints for the input layer
            output_connectivity: np.array of shape n_units x n_outputs with connectivity
                constraints for output layer
        """
        # Initialize base RNN class
        RNNBase.__init__(
            self,
            input_size=input_size,
            output_size=output_size,
            n_units=n_units,
            dale_ratio=dale_ratio,
            autopses=autopses,
            connectivity=connectivity,
            input_connectivity=input_connectivity,
            output_connectivity=output_connectivity,
        )

        # Define layers
        # basic RNN layer
        self.rnn_layer = nn.RNN(
            input_size=input_size,
            hidden_size=n_units,
            batch_first=True,
        )

        # linear output layer + activation function
        self.output_layer = nn.Linear(n_units, output_size)
        self.output_activation = nn.Tanh()

        # Freeze output layer's weights (don't train em)
        for p in self.output_layer.parameters():
            p.requires_grad = False

        # Build
        self.build()

    def get_recurrent_weights(self):
        """
        Get weights of recurrent layer
        """
        return npify(self.rnn_layer.weight_hh_l0)

    def set_recurrent_weights(self, weights):
        """
        Set weights of recurrent layer

        Arguments:
            weights (np.ndarray): (n_units * n_units) 2d array with recurrent weights
        """
        self.rnn_layer.weight_hh_l0 = nn.Parameter(torchify(weights))

    def forward(self, x, h=None):
        """
        Specifies how the forward dynamics of the network are computed

        Arguments:
            x (np.ndarray): network's input
            h (torch tensor) tensor with initial hidden state.
                If None the state is initialized as 0s.

        Returns:
            out (np.ndarray): the network's output
            h (torch tensor): updated hidden stae
        """
        # Get hidden state
        if h is None:
            h = self._initialize_hidden(x)

        # pass through RNN layer
        out, h = self.rnn_layer(x, h)

        # pass through linear layer
        out = self.output_activation(self.output_layer(out))
        return out, h
