import os
import numpy as np
import torch

import sys

sys.path.append("./")

from pyrnn import RNN
from pyrnn.tasks.three_bit_memory import (
    ThreeBitDataset,
    is_win,
    make_batch,
)
from pyrnn.analysis import (
    FixedPoints,
    FixedPointsConnectivity,
    list_fixed_points,
)
from pyrnn.render import (
    render_fixed_points,
    render_fixed_points_connectivity_analysis,
    render_fixed_points_connectivity_graph,
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# ----------------------------------- setup ---------------------------------- #
EXTRACT = False
CONNECTIVITY = False
RENDER = True

N = 2048
batch_size = 128

constant_inputs = [
    torch.from_numpy(np.array([0, 0, 0]).astype(np.float32)).reshape(1, 1, -1),
]

rnn = RNN.load("./3bit_memory.pt", n_units=64, input_size=3, output_size=3)

dataloader = torch.utils.data.DataLoader(
    ThreeBitDataset(N, dataset_length=batch_size),
    batch_size=batch_size,
    num_workers=0 if is_win else 2,
    shuffle=True,
    worker_init_fn=lambda x: np.random.seed(),
)
X, Y = list(dataloader)[0]

# ----------------------------- Find fixed points ---------------------------- #
if EXTRACT:
    o, h = rnn.predict_with_history(X)

    fp_finder = FixedPoints(rnn, speed_tol=1e-02, noise_scale=2)

    fp_finder.find_fixed_points(
        h,
        constant_inputs,
        n_initial_conditions=150,
        max_iters=9000,
        lr_decay_epoch=1500,
        max_fixed_points=27,
        gamma=0.1,
    )

    fp_finder.save_fixed_points("./3bit_fps.json")

# ----------------------------------- Plot ----------------------------------- #
fps = FixedPoints.load_fixed_points("./3bit_fps.json")
list_fixed_points(fps)
if RENDER:
    X, Y = make_batch(6000)
    _, _h = rnn.predict_with_history(X)

    render_fixed_points(
        _h, fps, alpha=0.005, scale=1, lw=0.2, sequential=False
    )

# ----------------------------- fps connectivity ----------------------------- #
if CONNECTIVITY:
    fps_connectivity = FixedPointsConnectivity(
        rnn,
        fps,
        n_initial_conditions_per_fp=128,
        noise_scale=0.1,
    )
    outcomes, graph = fps_connectivity.get_connectivity(
        constant_inputs[0], max_iters=1024
    )

if RENDER and CONNECTIVITY:
    render_fixed_points_connectivity_analysis(
        _h,
        fps,
        outcomes,
        alpha=0.005,
        sequential=True,
        traj_alpha=0.05,
        traj_radius=0.08,
        initial_conditions_radius=0.08,
    )

    render_fixed_points_connectivity_graph(_h, fps, graph, alpha=0.005)
