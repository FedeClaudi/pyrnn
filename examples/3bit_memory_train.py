import matplotlib.pyplot as plt
import os

# import sys

# sys.path.append("./")

from pyrnn import RNN
from pyrnn.plot import plot_training_loss
from pyrnn.tasks.three_bit_memory import (
    ThreeBitDataset,
    plot_predictions,
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# ---------------------------------- Params ---------------------------------- #
FIT = True

n_units = 64
N = 100
batch_size = 256
epochs = 700
lr_milestones = [500, 800]
lr = 0.005

# ---------------------------------- Fit RNN --------------------------------- #

dataset = ThreeBitDataset(N, dataset_length=8)

rnn = RNN(
    input_size=3,
    output_size=3,
    autopses=False,
    dale_ratio=0.8,
    n_units=n_units,
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
    report_path="./3bit_memory.txt",
)
rnn.save("./3bit_memory.pt")

plot_predictions(rnn, N, batch_size)
plot_training_loss(loss_history)
plt.show()
