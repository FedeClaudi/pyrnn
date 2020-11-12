import os
from vedo.colors import colorMap

# import sys

# sys.path.append("./")

from pyrnn import RNN
from pyrnn.render import render_state_history_pca_3d

# from pyrnn.plot import plot_render_state_history_pca_2d
from pyrnn.tasks.integrator import make_batch

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# ---------------------------------- Params ---------------------------------- #
FIT = True

K = 1
n_units = 128

rnn = RNN.load(
    "./integrator.pt",
    n_units=n_units,
    input_size=K,
    output_size=K,
)

# ------------------------------- Activity PCA ------------------------------- #
N = 2000
X, Y = make_batch(N, batch_size=2, k=K, switch_prob=0.1)
o, h = rnn.predict_with_history(X)


col = colorMap(X[0, :, 0], name="bwr", vmin=-0.3, vmax=0.3)
color = [col, col]
render_state_history_pca_3d(
    h,
    lw=0.2,
    alpha=0.01,
    mark_start=False,
    _show=True,
    color=color,
    color_by_trial=True,
)
