import matplotlib.pyplot as plt
import os

from pyrnn import CustomRNN, plot_training_loss, plot_state_history_pca_3d
from pyrnn.tasks.integrator import (
    IntegratorDataset,
    plot_predictions,
    make_batch,
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# ---------------------------------- Params ---------------------------------- #
FIT = True

K = 1
n_units = 64
N = 40
batch_size = 5000
epochs = 5000
lr_milestones = [600, 1300]
lr = 0.001
stop_loss = 0.001

# ------------------------------- Fit/load RNN ------------------------------- #
if FIT:
    dataset = IntegratorDataset(N, dataset_length=200, k=K)

    rnn = CustomRNN(
        input_size=K,
        output_size=K,
        autopses=True,
        dale_ratio=None,
        n_units=n_units,
    )

    rnn.save_params("integrator.json")

    loss_history = rnn.fit(
        dataset,
        N,
        n_epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        input_length=N,
        lr_milestones=lr_milestones,
        l2norm=0,
        report_path="3bit_memory.txt",
        stop_loss=stop_loss,
    )
    rnn.save("3bit_memory.pt")

    plot_predictions(rnn, N, batch_size, k=K)
    plot_training_loss(loss_history)
    plt.show()
else:
    rnn = CustomRNN.load(
        "3bit_memory.pt", n_units=n_units, input_size=K, output_size=K
    )

# ------------------------------- Activity PCA ------------------------------- #
actors = []
N = 15000
X, Y = make_batch(N, k=K)
o, h = rnn.predict_with_history(X)
plot_state_history_pca_3d(
    h, alpha=0.005, actors=actors, mark_start=True, axes=8
)
