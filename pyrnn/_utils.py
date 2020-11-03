import numpy as np
import torch
from itertools import combinations_with_replacement as combinations


def flatten_h(h):
    if len(h.shape) < 3:
        return h
    else:
        return h.reshape(-1, h.shape[-1])


def torchify(arr, flatten=True):
    tensor = torch.from_numpy(arr.astype(np.float32))

    if flatten:
        tensor.reshape(1, 1, len(arr))
    return tensor


def npify(tensor, flatten=True):
    if not isinstance(tensor, np.ndarray):
        tensor = tensor.detach().numpy()

    if flatten:
        ndims = len(tensor.shape)
        if ndims == 3:
            return tensor[0, 0, :]
        elif ndims == 2:
            return tensor[0, :]
        else:
            return tensor.ravel()
    else:
        return tensor


def prepend_dim(arr):
    return arr.reshape(1, -1)


def pairs(iterable):
    """
    returns all ordered pairs of items from an iterable
    """
    combos = list(combinations(iterable, 2))
    return combos + [(b, a) for a, b in combos]
