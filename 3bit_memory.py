import numpy as np
from numpy import random as rnd
import matplotlib.pyplot as plt
import torch
import os
from pyrnn import RNN, plot_training_loss

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


"""
    3 bit memory task
        input is a 3xN tensor with (0, 1, -1) values
        output is a 3xd tensor with (1, -1) values

    each input corresponds to noe output, the output
    is a `memory` of which state the input is in (1, -1)
"""

# ---------------------------------- Params ---------------------------------- #

N = 100
batch_size = 64
epochs = 2
lr = 0.01


# ----------------------------- batch making func ---------------------------- #


def make_batch():
    X_batch = torch.zeros((batch_size, N, 3))
    Y_batch = torch.zeros((batch_size, N, 3))
    for batch in range(batch_size):
        for m in range(3):
            # Define input
            X = torch.zeros(N)
            flips = (rnd.uniform(1, N - 1, int(N / 20))).astype(np.int32)
            flips2 = (rnd.uniform(1, N - 1, int(N / 20))).astype(np.int32)

            X[flips] = 1
            X[flips2] = -1
            X[0] = 1

            # Get correct output
            Y = torch.zeros(N)
            state = 0
            for n, x in enumerate(X):
                if x == 1:
                    state = 1
                elif x == -1:
                    state = -1

                Y[n] = state

            # RNN input: batch size * seq len * n_input
            X = X.reshape(1, N, 1)

            # out shape = (batch, seq_len, num_directions * hidden_size)
            # h_n shape  = (num_layers * num_directions, batch, hidden_size)
            Y = Y.reshape(1, N, 1)

            X_batch[batch, :, m] = X.squeeze()
            Y_batch[batch, :, m] = Y.squeeze()

    return X_batch, Y_batch


# ---------------------------------- Fit RNN --------------------------------- #

rnn = RNN(input_size=3, output_size=3)

loss_history = rnn.fit(
    make_batch, n_epochs=epochs, lr=lr, batch_size=batch_size, input_length=N
)
plot_training_loss(loss_history)

# ----------------------- Plot performance on new batch ---------------------- #

X, Y = make_batch()
o, h = rnn(X[0, :, :].unsqueeze(0), torch.zeros((1, 1, 50)))
o = o.detach().numpy()

f, axarr = plt.subplots(nrows=3)
for n, ax in enumerate(axarr):
    ax.plot(X[0, :, n], lw=4, color="k")
    ax.plot(Y[0, :, n], ls="--")
    ax.plot(o[0, :, n], lw=1, color="r")
    ax.set(title=f"Input {n}")
f.tight_layout()
# plt.plot(h.detach().numpy()[0, :], ls='--', lw=1, color='m')

plt.show()
print(h.shape)
