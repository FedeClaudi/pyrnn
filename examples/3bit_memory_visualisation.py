import os

# import sys

# sys.path.append("./")

from pyrnn import RNN
from pyrnn.render import render_state_history_pca_3d
from pyrnn.tasks.three_bit_memory import make_batch

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

n_units = 64
N = 5000
rnn = RNN.load(
    "./3bit_memory.pt", n_units=n_units, input_size=3, output_size=3
)

X, Y = make_batch(N)
o, h = rnn.predict_with_history(X)

render_state_history_pca_3d(h, alpha=0.01, lw=0.3)
