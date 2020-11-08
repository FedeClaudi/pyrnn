import numpy as np
from numpy import random as rnd
import matplotlib.pyplot as plt
import torch
from myterial import salmon, light_green_dark, indigo_light
from pyrnn._plot import clean_axes
import torch.utils.data as data
import sys

from pyrnn._utils import torchify

"""
    Sine wave task
        input is a 1xN constant tensor with values in range (0, 1)
        output is a 1xN tensor a sine wave whose frequency is specified by the input
"""

is_win = sys.platform == "win32"


class SineWaveDataset(data.Dataset):
    """
    creates a pytorch dataset for loading
    the data during training.
    """

    def __init__(
        self, sequence_length=50, dataset_length=1, min_freq=300, max_freq=310
    ):
        self.sequence_length = sequence_length
        self.dataset_length = dataset_length
        self.min_freq = min_freq
        self.max_freq = max_freq

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, item):
        freq_range = self.max_freq - self.min_freq

        inp = rnd.uniform(0, 10)

        x = np.ones(self.sequence_length) * inp
        X = torchify(x.astype(np.float32)).reshape(-1, 1)

        step = 0.01
        t = np.arange(0, self.sequence_length * step, step=step)

        y = 0.5 * np.sin(freq_range * inp * t)

        Y = torchify(y.astype(np.float32)).reshape(-1, 1)

        return X, Y


def make_batch(seq_len, freq=None):
    """
    Return a single batch of given length
    """
    dataloader = torch.utils.data.DataLoader(
        SineWaveDataset(seq_len, dataset_length=1),
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
