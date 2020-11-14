import numpy as np
from numpy import random as rnd
import matplotlib.pyplot as plt
import torch
from myterial import salmon, light_green_dark, indigo_light
import torch.utils.data as data
import sys
from random import choice
import itertools

from pyrnn._plot import clean_axes
from pyrnn._utils import torchify

"""
    2d memory task
        input is a 2xN tensor with random samples from a standard normal distribution
        output is a 2xN tensor with integrated values of the inputs

"""

is_win = sys.platform == "win32"


class IntegratorDataset(data.Dataset):
    """
    creates a pytorch dataset for loading
    the data during training.
    """

    speeds = (-0.3, 0.3)

    def __init__(
        self, sequence_length, dataset_length=1, k=2, switch_prob=0.05
    ):
        self.sequence_length = sequence_length
        self.dataset_length = dataset_length
        self.k = k
        self.switch_prob = switch_prob

    def __len__(self):
        return self.dataset_length

    def _mk(self):
        seq_len = self.sequence_length
        x = np.zeros(seq_len)

        speed = choice(self.speeds)
        for n in np.arange(seq_len):
            if rnd.rand() < self.switch_prob:
                speed = choice(self.speeds)
            x[n] = speed

        y = np.cumsum(x)
        phases = (np.arctan2(np.sin(y), np.cos(y))) / np.pi

        # add noise
        x += rnd.normal(scale=0.01, size=len(x))
        return torchify(x), torchify(phases)

    def __getitem__(self, item):
        seq_len = self.sequence_length
        if self.k > 1:
            X_batch = torch.zeros((seq_len, self.k))
            Y_batch = torch.zeros((seq_len, self.k))

            for m in range(self.k):
                x, y = self._mk()
                X_batch[:, m] = x
                Y_batch[:, m] = y

            return X_batch, Y_batch
        else:
            x, y = self._mk(item)
            return x.reshape(-1, 1), y.reshape(-1, 1)


def make_batch(seq_len, batch_size=1, **kwargs):
    """
    Return a single batch of given length
    """
    dataloader = torch.utils.data.DataLoader(
        IntegratorDataset(seq_len, dataset_length=batch_size, **kwargs),
        batch_size=batch_size,
        num_workers=0 if is_win else 2,
        shuffle=True,
        worker_init_fn=lambda x: np.random.seed(),
    )

    batch = [b for b in dataloader][0]
    return batch


def span_inputs(seq_len, k):
    """
    Creates a batched input
    that spans all combinations
    of inputs to the RNN
    (with no noise and no switching)
    """
    speeds = IntegratorDataset.speeds
    all = [speeds for i in range(k)]

    # Get all possible combinations
    combs = list(itertools.product(*all))

    X = np.zeros((len(combs), seq_len, k))
    for n, (s0, s1) in enumerate(combs):
        X[n, :, :] = np.vstack(
            [np.repeat(s0, seq_len), np.repeat(s1, seq_len)]
        ).T

    return torchify(X)


def plot_predictions(model, seq_len, batch_size, **kwargs):
    """
    Run the model on a single batch and plot
    the model's prediction's against the
    input data and labels.
    """
    X, Y = make_batch(seq_len, **kwargs)
    o, h = model.predict(X)

    k = X.shape[-1]

    f, axarr = plt.subplots(nrows=k, figsize=(12, 9))
    if not isinstance(axarr, np.ndarray):
        axarr = [axarr]
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
