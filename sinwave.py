import matplotlib.pyplot as plt
import os
from vedo.colors import colorMap
from vedo import show
import numpy as np

from pyrnn import RNN, plot_training_loss, plot_state_history_pca_3d
from pyrnn.tasks.sinewave import (
    SineDataset,
    plot_predictions,
)
from pyrnn._utils import torchify

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# ---------------------------------- Params ---------------------------------- #
FIT = True

n_units = 200

N = 512 if FIT else 20000

batch_size = 64
epochs = 512
lr = 0.005

lr_milestones = [600]
l2norm = 0
stop_loss = 0.01


# ------------------------------- Fit/load RNN ------------------------------- #
if FIT:
    dataset = SineDataset(N, dataset_length=8)

    rnn = RNN(
        n_units=n_units,
        input_size=1,
        output_size=1,
        autopses=True,
        dale_ratio=None,
    )

    loss_history = rnn.fit(
        dataset,
        N,
        n_epochs=epochs,
        lr=lr,
        lr_milestones=lr_milestones,
        batch_size=batch_size,
        input_length=N,
        l2norm=l2norm,
        stop_loss=stop_loss,
    )

    plot_training_loss(loss_history)
    rnn.save("sinewave.pt")

    plot_predictions(rnn, N, batch_size)
    plt.show()
else:
    rnn = RNN.load("sinewave.pt", n_units=n_units, input_size=1, output_size=1)

# ------------------------------- Activity PCA ------------------------------- #
dataset = SineDataset(N, dataset_length=200)
low, high = dataset.frequency_range

actors = []
for freq in np.arange(low, high, 4):
    color = colorMap(freq, name="viridis", vmin=low, vmax=high)

    freq = freq * np.ones(N).reshape(1, -1, 1).astype(np.float32)
    o, h = rnn.predict_with_history(torchify(freq, flatten=False))

    p, a = plot_state_history_pca_3d(
        h[0, 2:, :],
        alpha=1,
        lw=15,
        _show=False,
        color=color,
        actors=actors,
        mark_start=True,
    )
    actors.extend(a)

print("ready")
show(actors)
