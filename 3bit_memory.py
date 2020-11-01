import matplotlib.pyplot as plt

import os
from rich import print
from sklearn.decomposition import PCA
from vedo import Lines, show


from pyrnn import RNN, plot_training_loss
from pyrnn.tasks.three_bit_memory import (
    make_batch,
    plot_predictions,
    predict_with_history,
)


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


# ---------------------------------- Params ---------------------------------- #

N = 2500
batch_size = 64
epochs = 500
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
    rnn.save("3bit.pt")

    plot_predictions(rnn, N, batch_size)
    plt.show()
else:
    rnn = RNN.load("3bit.pt", input_size=3, output_size=3)


# ------------------------------- Activity PCA ------------------------------- #

X, Y, o, h = predict_with_history(rnn, N, batch_size)

pc = PCA(n_components=3).fit_transform(h)

points = [[pc[i, :], pc[i + 1, :]] for i in range(len(pc) - 1)]
line = Lines(points).lw(30).alpha(0.01).c("k")

print("ready")
show(line)
