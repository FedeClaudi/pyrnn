import matplotlib.pyplot as plt
import os

from pyrnn import RNN, plot_training_loss, plot_state_history_pca_3d
from pyrnn.tasks.three_bit_memory import (
    ThreeBitDataset,
    plot_predictions,
    make_batch,
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# ---------------------------------- Params ---------------------------------- #

N = 300
batch_size = 256
epochs = 350
lr = 0.005

FIT = True


# ------------------------------- Fit/load RNN ------------------------------- #
if __name__ == "__main__":
    if FIT:
        dataset = ThreeBitDataset(N, dataset_length=5)

        rnn = RNN(input_size=3, output_size=3)
        loss_history = rnn.fit(
            dataset,
            N,
            n_epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            input_length=N,
            autopses=False,
        )
        plot_training_loss(loss_history)
        rnn.save("test.pt")

        plot_predictions(rnn, N, batch_size)
        plt.show()
    else:
        rnn = RNN.load("3bit_fully_trained.pt", input_size=3, output_size=3)

    # ------------------------------- Activity PCA ------------------------------- #
    X, Y = make_batch(N)
    o, h = rnn.predict_with_history(X)

    plot_state_history_pca_3d(h, alpha=0.01)
