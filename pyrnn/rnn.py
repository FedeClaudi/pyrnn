import torch
import torch.nn as nn

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

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=n_units,
            batch_first=True,
        )
        self.output_layer = nn.Linear(n_units, output_size)
        self.output_activation = nn.Tanh()

    def forward(self, x, h):
        out, h = self.rnn(x, h)

        out = self.output_activation(self.output_layer(out))
        return out, h

    def fit(
        self,
        batch_maker,
        batch_size=64,
        n_epochs=100,
        lr=0.001,
        input_length=100,
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
            for epoch in range(n_epochs):
                X_batch, Y_batch = batch_maker()

                # Loop over batch samples
                batch_loss = 0
                for batchn in range(batch_size):
                    # initialise
                    X, Y = X_batch[batchn], Y_batch[batchn]
                    h = torch.zeros((1, 1, 50))

                    # zero gradient
                    optimizer.zero_grad()

                    # predict
                    output, h = self(X.reshape(1, input_length, -1), h)

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
