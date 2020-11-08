import matplotlib.pyplot as plt
import os
from vedo import show
import numpy as np
import torch
from vedo.colors import colorMap

from pyrnn import CustomRNN, plot_training_loss, plot_state_history_pca_3d
from pyrnn.tasks.sinewave import (
    SineWaveDataset,
    plot_predictions,
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# ---------------------------------- Params ---------------------------------- #
FIT = True

n_units = 64
N = 50
batch_size = 4000
epochs = 1024
lr_milestones = [500, 800]
lr = 0.001

# ------------------------------- Fit/load RNN ------------------------------- #
if FIT:
    dataset = SineWaveDataset(N, dataset_length=250)

    rnn = CustomRNN(
        input_size=1,
        output_size=1,
        autopses=True,
        dale_ratio=None,
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
        report_path="sinewave.txt",
        stop_loss=0.0007,
    )
    rnn.save("sinewave.pt")

    plot_predictions(rnn, N, batch_size)
    plot_training_loss(loss_history)
    plt.show()
else:
    rnn = CustomRNN.load(
        "sinewave.pt", n_units=n_units, input_size=1, output_size=1
    )

# ------------------------------- Activity PCA ------------------------------- #
actors = []
N = 500
for freq in np.arange(0.1, 1, step=0.1):
    X = torch.ones(1, N, 1) * freq
    o, h = rnn.predict_with_history(X)

    color = colorMap(freq, name="viridis", vmin=0, vmax=1)

    pca, actors = plot_state_history_pca_3d(
        h, alpha=1, color=color, actors=actors, _show=False, mark_start=True
    )
print("Ready")
show(*actors)
