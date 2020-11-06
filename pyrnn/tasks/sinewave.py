import numpy as np
from numpy import random as rnd
import matplotlib.pyplot as plt
import torch
from myterial import salmon, light_green_dark, purple_light
from pyrnn._plot import clean_axes
import torch.utils.data as data
import sys

from pyrnn._utils import npify, torchify

"""
    Sine wave task
        input is a 1xN tensor with value K
        output is a 1xN tensor with a sine wave of K

    each input corresponds to noe output, the output
    is a `memory` of which state the input is in (1, -1)
"""
is_win = sys.platform == "win32"


class SineDataset(data.Dataset):
    def __init__(
        self, sequence_length, dataset_length=1, frequency_range=(10, 15)
    ):
        self.sequence_length = sequence_length
        self.frequency_range = frequency_range
        self.dataset_length = dataset_length

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, item):
        freq = rnd.randint(self.frequency_range[0], self.frequency_range[1])
        t = np.arange(0, self.sequence_length * 0.01, step=0.01)

        x = freq + freq * np.sin(freq * t)
        x[10:] = freq
        X = torchify(x).reshape(-1, 1)

        y = 0.5 * np.sin(freq * t)
        Y = torchify(y.astype(np.float32)).reshape(-1, 1)

        return X, Y


def make_batch(seq_len, batch_len=1):
    dataloader = torch.utils.data.DataLoader(
        SineDataset(seq_len, dataset_length=batch_len),
        batch_size=batch_len,
        num_workers=0 if is_win else 2,
        shuffle=True,
        worker_init_fn=lambda x: np.random.seed(),
    )

    if batch_len == 1:
        batch = [b for b in dataloader][0]
    else:
        batch = list(dataloader)
    return batch


def plot_predictions(model, seq_len, batch_size):
    X, Y = make_batch(seq_len)
    o, h = model.predict(X)

    f, ax = plt.subplots(figsize=(12, 9))

    def t(tensor):
        arr = npify(tensor, flatten=False)
        return arr[0, :, 0]

    ax.plot(t(X), lw=3, color=salmon, ls="--", label="correct output")
    ax.plot(t(Y), lw=3, color=purple_light, ls="--", label="correct output")
    ax.plot(t(o), lw=2, color=light_green_dark, label="model output")
    ax.set(title=f"Input freq {t(X)[0]}")
    ax.legend()

    f.tight_layout()
    clean_axes(f)
