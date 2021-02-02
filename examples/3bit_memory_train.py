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
FIT = True

n_units = 64
N = 48  # trials length in dataset
batch_size = 256
epochs = 1000
lr_milestones = [1000, 2000]
lr = 50

# ---------------------------------- Fit RNN --------------------------------- #

dataset = ThreeBitDataset(N, dataset_length=256)

rnn = RNN(
    input_size=3,
    output_size=3,
    autopses=True,
    dale_ratio=None,
    n_units=n_units,
    on_gpu=False,
    w_in_train=True,
    w_out_train=True,
)

if FIT:
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

plot_predictions(rnn, N, batch_size)
plt.show()
