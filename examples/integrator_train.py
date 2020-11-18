import matplotlib.pyplot as plt
import os

import sys

sys.path.append("./")

from pyrnn import RNN
from pyrnn.plot import plot_training_loss
from pyrnn.tasks.integrator import (
    IntegratorDataset,
    plot_predictions,
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# ---------------------------------- Params ---------------------------------- #
K = 2
n_units = 64
N = 16
batch_size = 128
epochs = 5000  # 1024
dataset_length = 200
lr_milestones = None  # [x * dataset_length for x in [1000, 3000]]
lr = 0.005
stop_loss = 0.00025

# ------------------------------- Fit/load RNN ------------------------------- #
dataset = IntegratorDataset(
    N, dataset_length=dataset_length, k=K, switch_prob=0.01
)

rnn = RNN(
    input_size=K,
    output_size=K,
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
    report_path="integrator.txt",
    stop_loss=stop_loss,
)
rnn.save("integrator2.pt")

plot_predictions(rnn, N, batch_size, k=K)
plot_training_loss(loss_history)
plt.show()
