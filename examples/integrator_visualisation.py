import os
from vedo.colors import colorMap

import sys

sys.path.append("./")

from pyrnn import RNN
from pyrnn.render import render_state_history_pca_3d

# from pyrnn.plot import plot_render_state_history_pca_2d
from pyrnn.tasks.integrator import span_inputs

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# ---------------------------------- Params ---------------------------------- #
FIT = True

K = 2
n_units = 64

rnn = RNN.load(
    "./integrator2.pt",
    n_units=n_units,
    input_size=K,
    output_size=K,
)

# ------------------------------- Activity PCA ------------------------------- #
N = 1000
X = span_inputs(N, K)
o, h = rnn.predict_with_history(X)

col = colorMap(
    X[:, 0, 0] + 0.5 * X[:, 0, 1], name="viridis", vmin=-0.5, vmax=0.5
)
color = [col, col]
render_state_history_pca_3d(
    h,
    lw=0.02,
    alpha=0.1,
    mark_start=False,
    _show=True,
    color=col,
    color_by_trial=True,
)
