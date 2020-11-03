import os
import numpy as np
import torch


from pyrnn import RNN
from pyrnn.tasks.three_bit_memory import (
    make_batch,
)
from pyrnn.analysis import FixedPoints, FixedPointsConnectivity
from pyrnn.plot import (
    plot_fixed_points,
    plot_fixed_points_connectivity_analysis,
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# ----------------------------------- setup ---------------------------------- #
EXTRACT = False
N = 10000
batch_size = 64
epochs = 100
lr = 0.005


rnn = RNN.load("3bit_fully_trained.pt", input_size=3, output_size=3)

X, Y = make_batch(N)
o, h = rnn.predict_with_history(X)

constant_inputs = [
    torch.from_numpy(np.array([0, 0, 0]).astype(np.float32)).reshape(1, 1, -1),
]

# ----------------------------- Find fixed points ---------------------------- #
if EXTRACT:
    fp_finder = FixedPoints(
        rnn,
        speed_tol=1e-03,
        noise_scale=0.35,
    )

    fp_finder.find_fixed_points(
        h,
        constant_inputs,
        n_initial_conditions=1024,
        max_iters=2000,
        lr_decay_epoch=500,
        max_fixed_points=26,
    )

    fp_finder.save_fixed_points("fps.json")

# ----------------------------------- Plot ----------------------------------- #
fps = FixedPoints.load_fixed_points("fps.json")
plot_fixed_points(h, fps, alpha=0.01)

# ----------------------------- fps connectivity ----------------------------- #
fps_connectivity = FixedPointsConnectivity(rnn, fps, n_initial_conditions=200)
outcomes = fps_connectivity.get_connectivity(
    constant_inputs[0], max_iters=1024
)

plot_fixed_points_connectivity_analysis(h, fps, outcomes, alpha=0.01)
