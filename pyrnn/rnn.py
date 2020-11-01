import torch
import torch.nn as nn
from rich import print
from pyinspect._colors import mocassin, orange
import numpy as np
from rich.progress import track
from ._progress import train_progress


class RNN(nn.Module):
    def __init__(
        self,
        input_size=1,
        output_size=1,
        n_units=50,
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

    def _show(self):
        print(self)

    def __str__(self):
        self._show()

    def save(self, path):
        if not path.endswith(".pt"):
            raise ValueError("Expected a path point to a .pt file")
        print(f"[{mocassin}]Saving model at: [{orange}]{path}")
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path, *args, **kwargs):
        if not path.endswith(".pt"):
            raise ValueError("Expected a path point to a .pt file")
        print(f"[{mocassin}]Loading model from: [{orange}]{path}")
        model = cls(*args, **kwargs)
        model.load_state_dict(torch.load(path))
        model.eval()
        return model

    def _initialize_hidden(self):
        return torch.zeros((1, 1, self.n_units))

    def forward(self, x, h=None):
        if h is None:
            h = self._initialize_hidden()
        out, h = self.rnn_layer(x, h)
        out = self.output_activation(self.output_layer(out))
        return out, h

    def predict_with_history(self, X):
        print(f"[{mocassin}]Predicting input step by step")
        seq_len = X.shape[1]
        h = None
        hidden_trace = np.zeros((seq_len, self.n_units))
        output_trace = np.zeros((seq_len, self.output_size))
        for step in track(range(seq_len)):
            o, h = self(X[0, step, :].reshape(1, 1, -1), h)
            hidden_trace[step, :] = h.detach().numpy()
            output_trace[step, :] = o.detach().numpy()

        return output_trace, hidden_trace

    def predict(self, X):
        o, h = self(X[0, :, :].unsqueeze(0))
        o = o.detach().numpy()
        h = h.detach().numpy()
        return o, h

    def fit(
        self,
        batch_maker,
        *args,
        batch_size=64,
        n_epochs=100,
        lr=0.001,
        input_length=100,
        **kwargs,
    ):
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()

        losses = []
        with train_progress as progress:
            tid = progress.add_task(
                "Training",
                start=True,
                total=n_epochs,
                loss=0,
                lr=0.005,
            )

            # loop over epochs
            for epoch in range(n_epochs + 1):
                X_batch, Y_batch = batch_maker(*args, **kwargs)

                # Loop over batch samples
                batch_loss = 0
                for batchn in range(batch_size):
                    # initialise
                    X, Y = X_batch[batchn], Y_batch[batchn]

                    # zero gradient
                    optimizer.zero_grad()

                    # predict
                    output, h = self(X.reshape(1, input_length, -1))

                    # backprop + optimizer
                    loss = criterion(output[0, :], Y)
                    loss.backward(retain_graph=True)
                    optimizer.step()

                    batch_loss += loss.item()

                train_progress.update(
                    tid,
                    completed=epoch,
                    loss=batch_loss / batch_size,
                    lr=lr,
                )
                losses.append(batch_loss / batch_size)
        return losses
