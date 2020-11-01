import numpy as np
from numpy import random as rnd
import matplotlib.pyplot as plt
from rich.progress import track
import torch
from pyinspect._colors import salmon, dimgreen, lilla
from pyrnn._plot import clean_axes

"""
    3 bit memory task
        input is a 3xN tensor with (0, 1, -1) values
        output is a 3xd tensor with (1, -1) values

    each input corresponds to noe output, the output
    is a `memory` of which state the input is in (1, -1)
"""


def make_batch(seq_len, batch_size):
    X_batch = torch.zeros((batch_size, seq_len, 3))
    Y_batch = torch.zeros((batch_size, seq_len, 3))
    for batch in range(batch_size):
        for m in range(3):
            # Define input
            X = torch.zeros(seq_len)
            Y = torch.zeros(seq_len)

            flips = (rnd.uniform(1, seq_len - 1, int(seq_len / 20))).astype(
                np.int32
            )
            flips2 = (rnd.uniform(1, seq_len - 1, int(seq_len / 20))).astype(
                np.int32
            )

            X[flips] = 1
            X[flips2] = -1
            X[0] = 1

            # Get correct output
            state = 0
            for n, x in enumerate(X):
                if x == 1:
                    state = 1
                elif x == -1:
                    state = -1

                Y[n] = state

            # RNN input: batch size * seq len * n_input
            X = X.reshape(1, seq_len, 1)

            # out shape = (batch, seq_len, num_directions * hidden_size)
            # h_n shape  = (num_layers * num_directions, batch, hidden_size)
            Y = Y.reshape(1, seq_len, 1)

            X_batch[batch, :, m] = X.squeeze()
            Y_batch[batch, :, m] = Y.squeeze()

    return X_batch, Y_batch


def predict_with_history(model, seq_len, batch_size):
    X, Y = make_batch(seq_len, batch_size)

    h = None
    hidden_trace = np.zeros((seq_len, model.n_units))
    output_trace = np.zeros((seq_len, model.output_size))
    for step in track(range(seq_len)):
        o, h = model(X[0, step, :].reshape(1, 1, -1), h)
        hidden_trace[step, :] = h.detach().numpy()
        output_trace[step, :] = o.detach().numpy()

    return X, Y, hidden_trace, output_trace


def predict(model, seq_len, batch_size):
    X, Y = make_batch(seq_len, batch_size)

    o, h = model(X[0, :, :].unsqueeze(0))
    o = o.detach().numpy()
    h = h.detach().numpy()
    return X, Y, o, h


def plot_predictions(model, seq_len, batch_size):
    X, Y, o, h = predict(model, seq_len, batch_size)

    f, axarr = plt.subplots(nrows=3, figsize=(12, 9))
    for n, ax in enumerate(axarr):
        ax.plot(X[0, :, n], lw=2, color=salmon, label="input")
        ax.plot(Y[0, :, n], lw=3, color=lilla, ls="--", label="correct output")
        ax.plot(o[0, :, n], lw=2, color=dimgreen, label="model output")
        ax.set(title=f"Input {n}")
        ax.legend()

    f.tight_layout()
    clean_axes(f)
