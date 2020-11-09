import numpy as np
from numpy import random as rnd
import matplotlib.pyplot as plt
import torch
from myterial import salmon, light_green_dark, indigo_light
from pyrnn._plot import clean_axes
import torch.utils.data as data
import sys

from pyrnn._utils import torchify, npify

"""
    Sine wave task
        input is a 1xN constant tensor with values in range (0, 1)
        output is a 1xN tensor a sine wave whose frequency is specified by the input
"""

is_win = sys.platform == "win32"


def sinewave_on_batch_start(rnn, x, y):
    sequence_length = len(x)
    inp = npify(x).max()
    omega = 0.1 + round(0.5 * inp, 1)  # freq range (.1, .6) radians
    step = 0.01
    t = np.arange(0, sequence_length * step, step=step)

    x = np.sin(2 * np.pi * omega * t)
    X = torchify(x.astype(np.float32)).reshape(-1, 1)

    h = None
    for step in range(sequence_length):
        o, h = rnn(X[step, :].reshape(1, 1, -1), h)
    return h


class SineWaveDataset(data.Dataset):
    """
    creates a pytorch dataset for loading
    the data during training.
    """

    def __init__(
        self,
        sequence_length=500,
        dataset_length=1,
    ):
        self.sequence_length = sequence_length
        self.dataset_length = dataset_length

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, item):
        J = rnd.randint(0, 51)
        inp = J / 51  # + .25
        omega = 0.1 + round(0.5 * inp, 1)  # freq range (.1, .6) radians
        step = 0.01
        t = np.arange(0, self.sequence_length * step, step=step)

        x = np.ones(self.sequence_length) * inp

        X = torchify(x).reshape(-1, 1)

        y = np.sin(2 * np.pi * omega * t)
        Y = torchify(y).reshape(-1, 1)

        return X, Y


def make_batch(seq_len, freq=None, **kwargs):
    """
    Return a single batch of given length
    """
    dataloader = torch.utils.data.DataLoader(
        SineWaveDataset(seq_len, dataset_length=1, **kwargs),
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

    f, ax = plt.subplots(figsize=(12, 9))

    ax.plot(X[0, :, 0], lw=2, color=salmon, label="input")
    ax.plot(
        Y[0, :, 0],
        lw=3,
        color=indigo_light,
        ls="--",
        label="correct output",
    )
    ax.plot(o[0, :, 0], lw=2, color=light_green_dark, label="model output")
    ax.legend()

    f.tight_layout()
    clean_axes(f)


def plot_freqs():
    f, axarr = plt.subplots(nrows=10, figsize=(16, 9))
    for ax in axarr:
        X, Y = make_batch(500)
        ax.plot(
            X[0, :, 0],
            lw=3,
            color="orange",
            ls="--",
            label="input",
        )

        ax.plot(
            Y[0, :, 0],
            lw=3,
            color=indigo_light,
            ls="--",
            label="correct output",
        )
        ax.legend()
        ax.set(title=str(X.max()))


if __name__ == "__main__":
    plot_freqs()
