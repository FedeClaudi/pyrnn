import torch
import torch.nn as nn
from rich import print
from myterial import amber_light, orange
import numpy as np
from rich.progress import track
import sys

from ._progress import train_progress
from ._utils import npify, torchify


is_win = sys.platform == "win32"


class RecurrentWeightsInitializer(object):
    def __init__(
        self,
        initial_weights,
        dale_ratio=None,
        autopses=True,
        connectivity=None,
    ):
        self.weights = initial_weights
        self.connectivity = np.ones_like(self.weights)

        if connectivity is not None:
            self._set_connectivity(connectivity)

        if dale_ratio is not None:
            self._apply_dale_ratio(dale_ratio)

        if not autopses:
            self._remove_autopses()

    def _remove_autopses(self):
        np.fill_diagonal(self.weights, 0)
        return

    def _apply_dale_ratio(self, dale_ratio):
        if dale_ratio < 0 or dale_ratio > 1:
            raise ValueError(f"Invalid dale ratio value of: {dale_ratio}")

        n_units = len(self.weights)
        n_excitatory = int(np.floor(n_units * dale_ratio))

        dale_vec = np.ones(n_units)
        dale_vec[n_excitatory:] = -1

        self.weights = np.matmul(np.abs(self.weights), np.diag(dale_vec))

    def _set_connectivity(self, connectivity):
        # see: https://colab.research.google.com/github/murraylab/PsychRNN/blob/master/docs/notebooks/BiologicalConstraints.ipynb
        # and: https://github.com/murraylab/PsychRNN/blob/master/psychrnn/backend/initializations.py
        raise NotImplementedError(
            "Need to set up a method to enforce connectivity constraints"
        )


class RNN(nn.Module):
    r"""
    h_{t}=\tanh \left(W_{i h} x_{t}+b_{i h}+W_{h h} h_{(t-1)}+b_{h h}\right)
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

        super(RNN, self).__init__()

        self.n_units = n_units
        self.output_size = output_size
        self.input_size = input_size

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

        self._set_recurrent_weights(dale_ratio, autopses, connectivity)
        self._get_recurrent_weights()

    def _get_recurrent_weights(self):
        self.recurrent_weights = npify(
            self.rnn_layer.weight_hh_l0, flatten=False
        )

    def _set_recurrent_weights(self, dale_ratio, autopses, connectivity):
        iw = npify(self.rnn_layer.weight_hh_l0, flatten=False)
        initializer = RecurrentWeightsInitializer(
            iw,
            dale_ratio=dale_ratio,
            autopses=autopses,
            connectivity=connectivity,
        )
        self.rnn_layer.weight_hh_l0 = nn.Parameter(
            torchify(initializer.weights, flatten=False)
        )

    def _show(self):
        print(self)

    def __str__(self):
        self._show()

    def save(self, path):
        if not path.endswith(".pt"):
            raise ValueError("Expected a path point to a .pt file")
        print(f"[{amber_light}]Saving model at: [{orange}]{path}")
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path, *args, **kwargs):
        if not path.endswith(".pt"):
            raise ValueError("Expected a path point to a .pt file")

        print(f"[{amber_light}]Loading model from: [{orange}]{path}")
        model = cls(*args, **kwargs)
        model.load_state_dict(torch.load(path))
        model.eval()
        return model

    def _initialize_hidden(self, x, *args):
        return torch.zeros((1, x.shape[0], self.n_units))

    def forward(self, x, h=None):
        if h is None:
            h = self._initialize_hidden(x)

        out, h = self.rnn_layer(x, h)
        out = self.output_activation(self.output_layer(out))
        return out, h

    def fit(
        self,
        dataset,
        *args,
        batch_size=64,
        n_epochs=100,
        lr=0.001,
        lr_milestones=None,
        gamma=0.1,
        input_length=100,
        l2norm=0.0001,
        stop_loss=None,
        **kwargs,
    ):
        stop_loss = stop_loss or -1

        train_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0 if is_win else 2,
            shuffle=True,
            worker_init_fn=lambda x: np.random.seed(),
        )

        # Get optimizer
        optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=l2norm, amsgrad=True
        )

        # Set up leraning rate scheudler
        lr_milestones = lr_milestones or [100000000]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=lr_milestones, gamma=gamma
        )

        # Set up training loss
        criterion = torch.nn.MSELoss()

        losses = []
        with train_progress as progress:
            tid = progress.add_task(
                "Training",
                start=True,
                total=n_epochs,
                loss=0,
                lr=lr,
            )

            # loop over epochs
            for epoch in range(n_epochs + 1):
                # Loop over batch samples
                batch_loss = 0
                for batchn, batch in enumerate(train_dataloader):
                    # initialise
                    X, Y = batch
                    self._labels = Y

                    # zero gradient
                    optimizer.zero_grad()

                    # predict
                    output, h = self(X)

                    # backprop + optimizer
                    loss = criterion(output, Y)
                    loss.backward(retain_graph=True)

                    optimizer.step()
                    scheduler.step()

                    # get current lr
                    lr = scheduler.get_last_lr()[-1]

                    batch_loss += loss.item()

                train_progress.update(
                    tid,
                    completed=epoch,
                    loss=batch_loss,
                    lr=lr,
                )
                losses.append(batch_loss)

                if batch_loss <= stop_loss:
                    break
        return losses

    def predict_with_history(self, X):
        print(f"[{amber_light}]Predicting input step by step")
        seq_len = X.shape[1]
        n_trials = X.shape[0]

        hidden_trace = np.zeros((n_trials, seq_len, self.n_units))
        output_trace = np.zeros((n_trials, seq_len, self.output_size))

        for trialn in range(n_trials):
            h = None
            for step in track(
                range(seq_len), description=f"trial {trialn}/{n_trials}"
            ):
                o, h = self(X[trialn, step, :].reshape(1, 1, -1), h)
                hidden_trace[trialn, step, :] = h.detach().numpy()
                output_trace[trialn, step, :] = o.detach().numpy()

        return output_trace, hidden_trace

    def predict(self, X):
        o, h = self(X[0, :, :].unsqueeze(0))
        o = o.detach().numpy()
        h = h.detach().numpy()
        return o, h
