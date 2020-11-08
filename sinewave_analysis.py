import os
import numpy as np
import torch

from pyrnn import CustomRNN
from pyrnn.tasks.sinewave import (
    SineWaveDataset,
    is_win,
)
from pyrnn.analysis import (
    FixedPoints,
    FixedPointsConnectivity,
    list_fixed_points,
)
from pyrnn.plot import (
    plot_fixed_points,
    plot_fixed_points_connectivity_analysis,
    plot_fixed_points_connectivity_graph,
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# ----------------------------------- setup ---------------------------------- #
EXTRACT = True
CONNECTIVITY = False
RENDER = True

N = 2048 if EXTRACT else 512
batch_size = 128 if EXTRACT else 32


rnn = CustomRNN.load("3bit_memory.pt", n_units=50, input_size=3, output_size=3)

dataloader = torch.utils.data.DataLoader(
    SineWaveDataset(N, dataset_length=batch_size),
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
    fp_finder = FixedPoints(rnn, speed_tol=5e-03, noise_scale=1.75)

    fp_finder.find_fixed_points(
        h,
        constant_inputs,
        n_initial_conditions=100,
        max_iters=15000,
        lr_decay_epoch=1500,
        max_fixed_points=27,
        gamma=0.1,
    )

    fp_finder.save_fixed_points("rnn.json")

# ----------------------------------- Plot ----------------------------------- #
fps = FixedPoints.load_fixed_points("rnn.json")
list_fixed_points(fps)
if RENDER and EXTRACT:
    plot_fixed_points(h, fps, alpha=0.005, scale=1, sequential=False)

# ----------------------------- fps connectivity ----------------------------- #
if CONNECTIVITY:
    fps_connectivity = FixedPointsConnectivity(
        rnn,
        fps,
        n_initial_conditions_per_fp=64,
        noise_scale=0.1,
    )
    outcomes, graph = fps_connectivity.get_connectivity(
        constant_inputs[0], max_iters=1024
    )

if RENDER and CONNECTIVITY:
    plot_fixed_points_connectivity_analysis(
        h, fps, outcomes, alpha=0.005, sequential=True
    )

    plot_fixed_points_connectivity_graph(h, fps, graph, alpha=0.005)
