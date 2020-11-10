import os
from vedo.colors import colorMap
import numpy as np

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
N = 1200
X, Y = make_batch(N, batch_size=4, k=K, switch_prob=-1.0)
o, h = rnn.predict_with_history(X)

color = colorMap(np.arange(N), name="bwr", vmin=-1, vmax=1)
render_state_history_pca_3d(
    h,
    lw=0.4,
    alpha=0.01,
    mark_start=False,
    _show=True,
)
