import numpy as np
from numpy import random as rnd
import matplotlib.pyplot as plt
import torch
from myterial import salmon, light_green_dark, indigo_light
import torch.utils.data as data
import sys

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

    def __init__(self, sequence_length, dataset_length=1):
        self.sequence_length = sequence_length
        self.dataset_length = dataset_length

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, item):
        seq_len = self.sequence_length
        X_batch = torch.zeros((seq_len, 2))
        Y_batch = torch.zeros((seq_len, 2))

        for m in range(2):
            x = rnd.normal(0, 1, self.sequence_length)
            y = np.cumsum(x)

            fact = 1 / (y.max() - y.min())
            x *= fact
            y *= fact

            X_batch[:, m] = torchify(x)
            Y_batch[:, m] = torchify(y)

        return X_batch, Y_batch


def make_batch(seq_len):
    """
    Return a single batch of given length
    """
    dataloader = torch.utils.data.DataLoader(
        IntegratorDataset(seq_len, dataset_length=1),
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

    f, axarr = plt.subplots(nrows=2, figsize=(12, 9))
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
