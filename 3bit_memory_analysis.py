import os
import numpy as np
import torch

from pyrnn import RNN
from pyrnn.tasks.three_bit_memory import (
    ThreeBitDataset,
    is_win,
)
from pyrnn.analysis import FixedPoints, FixedPointsConnectivity
from pyrnn.plot import (
    plot_fixed_points,
    plot_fixed_points_connectivity_analysis,
    plot_fixed_points_connectivity_graph,
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# ----------------------------------- setup ---------------------------------- #
EXTRACT = True
RENDER = True

N = 512
batch_size = 32


rnn = RNN.load("test.pt", input_size=3, output_size=3)

dataloader = torch.utils.data.DataLoader(
    ThreeBitDataset(N, dataset_length=batch_size),
    batch_size=batch_size,
    num_workers=0 if is_win else 2,
    shuffle=True,
    worker_init_fn=lambda x: np.random.seed(),
)
X, Y = list(dataloader)[0]
o, h = rnn.predict_with_history(X)

constant_inputs = [
    torch.from_numpy(np.array([0, 0, 0]).astype(np.float32)).reshape(1, 1, -1),
]

# ----------------------------- Find fixed points ---------------------------- #
if EXTRACT:
    fp_finder = FixedPoints(
        rnn,
        speed_tol=3e-03,
        noise_scale=1.5,
    )

    fp_finder.find_fixed_points(
        h,
        constant_inputs,
        n_initial_conditions=256,
        max_iters=5000,
        lr_decay_epoch=1500,
        max_fixed_points=27,
    )

    fp_finder.save_fixed_points("fps2.json")

# ----------------------------------- Plot ----------------------------------- #
fps = FixedPoints.load_fixed_points("fps2.json")
if RENDER:
    plot_fixed_points(h, fps, alpha=0.005)

# ----------------------------- fps connectivity ----------------------------- #
fps_connectivity = FixedPointsConnectivity(
    rnn,
    fps,
    n_initial_conditions=12000,
    noise_scale=0.2,
)
outcomes, graph = fps_connectivity.get_connectivity(
    constant_inputs[0], max_iters=1024
)

if RENDER:
    plot_fixed_points_connectivity_analysis(h, fps, outcomes, alpha=0.005)

    plot_fixed_points_connectivity_graph(h, fps, graph, alpha=0.005)
