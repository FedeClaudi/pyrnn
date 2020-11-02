import os
import numpy as np
import torch


from pyrnn import RNN
from pyrnn.tasks.three_bit_memory import (
    make_batch,
)
from pyrnn.analysis import FixedPoints


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


N = 10000
batch_size = 64
epochs = 100
lr = 0.005


rnn = RNN.load("3bit_fully_trained.pt", input_size=3, output_size=3)

X, Y = make_batch(N, 1)
o, h = rnn.predict_with_history(X)

fp_finder = FixedPoints(
    rnn,
    # speed_tol=1e-03,
    speed_tol=1,
    noise_scale=0.35,
)

constant_inputs = [
    torch.from_numpy(np.array([0, 0, 0]).astype(np.float32)).reshape(1, 1, -1),
]

fps = fp_finder.find_fixed_points(
    h,
    constant_inputs,
    n_initial_conditions=500,
    max_iters=3000,
    lr_decay_epoch=500,
    max_fixed_points=1,
)

fp_finder.save_fixed_points("fps.json")
