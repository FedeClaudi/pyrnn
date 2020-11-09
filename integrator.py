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

n_units = 64
N = 256 if FIT else 15000
batch_size = 5000
epochs = 800
lr_milestones = [300, 1300]
lr = 0.01

# ------------------------------- Fit/load RNN ------------------------------- #
if FIT:
    dataset = IntegratorDataset(N, dataset_length=1000)

    # rnn = CustomRNN(
    #     input_size=2,
    #     output_size=2,
    #     autopses=True,
    #     dale_ratio=None,
    #     n_units=n_units,
    # )
    rnn = CustomRNN.from_json("integrator.json")

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
    )
    rnn.save("3bit_memory.pt")

    plot_predictions(rnn, N, batch_size)
    plot_training_loss(loss_history)
    plt.show()
else:
    rnn = CustomRNN.load(
        "3bit_memory.pt", n_units=n_units, input_size=2, output_size=2
    )

# ------------------------------- Activity PCA ------------------------------- #
actors = []
X, Y = make_batch(N)
o, h = rnn.predict_with_history(X)
plot_state_history_pca_3d(
    h, alpha=0.01, actors=actors, mark_start=True, axes=8
)
