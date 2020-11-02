import matplotlib.pyplot as plt
import os


from pyrnn import RNN, plot_training_loss, plot_state_history_pca_3d
from pyrnn.tasks.three_bit_memory import (
    make_batch,
    plot_predictions,
)


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


# ---------------------------------- Params ---------------------------------- #

N = 15000
batch_size = 64
epochs = 300
lr = 0.005

FIT = False


# ------------------------------- Fit/load RNN ------------------------------- #

if FIT:
    rnn = RNN(input_size=3, output_size=3)
    loss_history = rnn.fit(
        make_batch,
        N,
        batch_size,
        n_epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        input_length=N,
    )
    plot_training_loss(loss_history)
    rnn.save("3bit_fully_trained.pt")

    plot_predictions(rnn, N, batch_size)
    plt.show()
else:
    rnn = RNN.load("3bit_fully_trained.pt", input_size=3, output_size=3)

# ------------------------------- Activity PCA ------------------------------- #
X, Y = make_batch(N, 1)
o, h = rnn.predict_with_history(X)

plot_state_history_pca_3d(h, alpha=0.01)
