import matplotlib.pyplot as plt
import os

import sys

sys.path.append("./")

from pyrnn import CTRNN as RNN
from pyrnn.plot import plot_training_loss
from three_bit_memory import (
    ThreeBitDataset,
    plot_predictions,
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# ---------------------------------- Params ---------------------------------- #
FIT = False

n_units = 128
N = 48  # trials length in dataset
batch_size = 256
epochs = 5000
lr_milestones = [1000, 2000, 4000]
lr = 0.02

# ---------------------------------- Fit RNN --------------------------------- #

dataset = ThreeBitDataset(N, dataset_length=256)


if FIT:
    rnn = RNN(
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

    loss_history = rnn.fit(
        dataset,
        N,
        n_epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        input_length=N,
        lr_milestones=lr_milestones,
        l2norm=0,
    )
    rnn.save("./3bit_memory.pt")
    plot_training_loss(loss_history)
else:
    rnn = RNN.load(
        "./3bit_memory.pt",
        input_size=3,
        output_size=3,
        autopses=True,
        dale_ratio=None,
        n_units=n_units,
        on_gpu=FIT,
        w_in_train=True,
        w_out_train=True,
        tau=50,
    )
plot_predictions(rnn, N, batch_size)
plt.show()
