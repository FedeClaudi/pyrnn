import matplotlib.pyplot as plt
import os

import sys

sys.path.append("./")

from pyrnn import CTRNN as RNN
from three_bit_memory import (
    ThreeBitDataset,
    plot_predictions,
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# ---------------------------------- Params ---------------------------------- #
FIT = True

n_units = 64
N = 2048  # trials length in dataset
batch_size = 256
epochs = 2000
lr_milestones = None  # [1000, 2000, 5000, 6000]
lr = 0.0025

# ---------------------------------- Fit RNN --------------------------------- #

dataset = ThreeBitDataset(N, dataset_length=128)


if FIT:
    rnn = RNN(
        input_size=3,
        output_size=3,
        autopses=True,
        dale_ratio=None,
        n_units=n_units,
        on_gpu=False,
        w_in_train=True,
        w_out_train=True,
        tau=50,
        dt=5,
    )

    loss_history = rnn.fit(
        dataset,
        N,
        n_epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        input_length=N,
        lr_milestones=lr_milestones,
        l2norm=0,
        save_at_min_loss=True,
        save_path="./3bit_memory_minloss.pt",
    )
    rnn.save("./3bit_memory.pt")

    rnn = RNN.load(
        "./3bit_memory_minloss.pt",
        input_size=3,
        output_size=3,
        autopses=True,
        dale_ratio=None,
        n_units=n_units,
        on_gpu=FIT,
        w_in_train=True,
        w_out_train=True,
        tau=50,
        dt=5,
    )
else:
    rnn = RNN.load(
        "./3bit_memory_minloss.pt",
        input_size=3,
        output_size=3,
        autopses=True,
        dale_ratio=None,
        n_units=n_units,
        on_gpu=FIT,
        w_in_train=True,
        w_out_train=True,
        tau=50,
        dt=5,
    )
plot_predictions(rnn, N, batch_size)
plt.show()
