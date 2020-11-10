import matplotlib.pyplot as plt
import os
from vedo.colors import colorMap
from vedo import show

from pyrnn import RNN, plot_training_loss, plot_state_history_pca_3d
from pyrnn.tasks.integrator import (
    IntegratorDataset,
    plot_predictions,
    make_batch,
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# ---------------------------------- Params ---------------------------------- #
FIT = False

K = 1
n_units = 128
N = 20
batch_size = 128
epochs = 3000  # 1024
lr_milestones = [10000]
lr = 0.001
stop_loss = 0.0005

# ------------------------------- Fit/load RNN ------------------------------- #
if FIT:
    dataset = IntegratorDataset(N, dataset_length=200, k=K)

    rnn = RNN(
        input_size=K,
        output_size=K,
        autopses=True,
        dale_ratio=None,
        n_units=n_units,
        w_in_bias=True,
        w_in_train=True,
        w_out_bias=True,
        w_out_train=True,
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
    rnn.save("integrator.pt")

    plot_predictions(rnn, N, batch_size, k=K)
    plot_training_loss(loss_history)
    plt.show()
else:
    rnn = RNN.load(
        "integrator.pt",
        n_units=n_units,
        input_size=K,
        output_size=K,
        w_in_bias=True,
        w_in_train=True,
        w_out_bias=True,
        w_out_train=True,
    )

# ------------------------------- Activity PCA ------------------------------- #
actors = []
N = 100
for i in range(10):
    X, Y = make_batch(N, k=K, switch_prob=0.1)
    o, h = rnn.predict_with_history(X)

    color = colorMap(X[0, :, 0], name="bwr", vmin=-0.3, vmax=0.3)
    _, actors = plot_state_history_pca_3d(
        h,
        alpha=0.8,
        actors=actors,
        mark_start=False,
        color=color,
        _show=False,
    )
print("render ready")
show(*actors)
