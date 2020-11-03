import numpy as np
import torch


def torchify(arr):
    return torch.from_numpy(arr.astype(np.float32)).reshape(1, 1, len(arr))


def npify(tensor):
    if not isinstance(tensor, np.ndarray):
        tensor = tensor.detach().numpy()

    ndims = len(tensor.shape)
    if ndims == 3:
        return tensor[0, 0, :]
    elif ndims == 2:
        return tensor[0, :]
    else:
        return tensor.ravel()


def prepend_dim(arr):
    return arr.reshape(1, -1)
