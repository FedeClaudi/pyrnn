import os
from vedo.colors import colorMap

import sys

sys.path.append("./")

from pyrnn import RNN
from pyrnn.render import render_state_history_pca_3d

# from pyrnn.plot import plot_render_state_history_pca_2d
from pyrnn.tasks.integrator import make_batch

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
N = 2000
# X = span_inputs(N, K)
X, _ = make_batch(N, 1, k=K, switch_prob=0.01)
o, h = rnn.predict_with_history(X)

# col = colorMap(
#     X[:, 0, 0] + 0.5 * X[:, 0, 1], name="viridis", vmin=-0.5, vmax=0.5
# )
color = [
    colorMap(X[n, :, 0] + 0.5 * X[n, :, 1], name="tab20", vmin=-0.5, vmax=0.5)
    for n in range(1)
]
render_state_history_pca_3d(
    h,
    lw=0.1,
    alpha=0.1,
    mark_start=False,
    _show=True,
    color=color,
    color_by_trial=True,
)
