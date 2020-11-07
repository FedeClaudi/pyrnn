import numpy as np
from numpy import random as rnd
import matplotlib.pyplot as plt
import torch
from myterial import salmon, light_green_dark, indigo_light
from pyrnn._plot import clean_axes
import torch.utils.data as data
import sys

"""
    3 bit memory task
        input is a 3xN tensor with (0, 1, -1) values
        output is a 3xN tensor with (1, -1) values

    each input corresponds to noe output, the output
    is a `memory` of which state the input is in (1, -1)

    ThreeBitDataset creates a pytorch dataset for loading
    the data during training.
"""

is_win = sys.platform == "win32"


class ThreeBitDataset(data.Dataset):
    """
    creates a pytorch dataset for loading
    the data during training.
    """

    def __init__(self, sequence_length, dataset_length=1):
        self.sequence_length = sequence_length
        self.dataset_length = dataset_length

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, item):
        seq_len = self.sequence_length
        X_batch = torch.zeros((seq_len, 3))
        Y_batch = torch.zeros((seq_len, 3))

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
            Y = Y.reshape(1, seq_len, 1)

            X_batch[:, m] = X.squeeze()
            Y_batch[:, m] = Y.squeeze()

        return X_batch, Y_batch


def make_batch(seq_len):
    """
    Return a single batch of given length
    """
    dataloader = torch.utils.data.DataLoader(
        ThreeBitDataset(seq_len, dataset_length=1),
        batch_size=1,
        num_workers=0 if is_win else 2,
        shuffle=True,
        worker_init_fn=lambda x: np.random.seed(),
    )

    batch = [b for b in dataloader][0]
    return batch


def plot_predictions(model, seq_len, batch_size):
    """
    Run the model on a single batch and plot
    the model's prediction's against the
    input data and labels.
    """
    X, Y = make_batch(seq_len)
    o, h = model.predict(X)

    f, axarr = plt.subplots(nrows=3, figsize=(12, 9))
    for n, ax in enumerate(axarr):
        ax.plot(X[0, :, n], lw=2, color=salmon, label="input")
        ax.plot(
            Y[0, :, n],
            lw=3,
            color=indigo_light,
            ls="--",
            label="correct output",
        )
        ax.plot(o[0, :, n], lw=2, color=light_green_dark, label="model output")
        ax.set(title=f"Input {n}")
        ax.legend()

    f.tight_layout()
    clean_axes(f)
