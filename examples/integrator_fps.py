import os
import numpy as np
import torch

import sys

sys.path.append("./")

from pyrnn import RNN
from pyrnn.tasks.integrator import (
    IntegratorDataset,
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
from pyrnn._utils import torchify

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# ----------------------------------- setup ---------------------------------- #
EXTRACT = True
CONNECTIVITY = True
RENDER = True

N = 200
batch_size = 64
K = 2
n_units = 64

constant_inputs = [
    torchify(np.zeros(K)).reshape(1, 1, -1),
]

rnn = RNN.load(
    "./integrator2.pt",
    n_units=n_units,
    input_size=K,
    output_size=K,
)

dataloader = torch.utils.data.DataLoader(
    IntegratorDataset(N, dataset_length=batch_size, k=K, switch_prob=0.01),
    batch_size=batch_size,
    num_workers=0 if is_win else 2,
    shuffle=True,
    worker_init_fn=lambda x: np.random.seed(),
)
X, Y = list(dataloader)[0]

# ----------------------------- Find fixed points ---------------------------- #
if EXTRACT:
    o, h = rnn.predict_with_history(X)

    fp_finder = FixedPoints(rnn, speed_tol=8e-02, noise_scale=2)

    fp_finder.find_fixed_points(
        h,
        constant_inputs,
        n_initial_conditions=150,
        max_iters=9000,
        lr_decay_epoch=1500,
        max_fixed_points=2,
        gamma=0.1,
    )

    fp_finder.save_fixed_points("./integrator_fps.json")

# ----------------------------------- Plot ----------------------------------- #
fps = FixedPoints.load_fixed_points("./integrator_fps.json")
list_fixed_points(fps)
if RENDER:
    X, Y = make_batch(1000, batch_size=3, k=K)
    _, _h = rnn.predict_with_history(X)

    render_fixed_points(
        _h, fps, alpha=0.005, scale=1, lw=0.4, sequential=False
    )

# ----------------------------- fps connectivity ----------------------------- #
if CONNECTIVITY:
    fps_connectivity = FixedPointsConnectivity(
        rnn,
        fps,
        n_initial_conditions_per_fp=24,
        noise_scale=0.5,
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
        sequential=False,
        traj_alpha=0.8,
        traj_radius=0.3,
        initial_conditions_radius=0.3,
    )

    render_fixed_points_connectivity_graph(_h, fps, graph, alpha=0.005)
