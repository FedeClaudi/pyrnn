import numpy as np
import torch
from itertools import combinations_with_replacement as combinations
import signal


def flatten_h(h):
    """
    Flatten the hidden state array
    """
    if len(h.shape) < 3:
        return h
    else:
        return h.reshape(-1, h.shape[-1])


def torchify(arr, flatten=False):
    """
    Turn a numpy array into a tensor
    """
    tensor = torch.from_numpy(arr.astype(np.float32))

    if flatten:
        tensor.reshape(1, 1, len(arr))
    return tensor


def npify(tensor, flatten=False):
    """
    Turn a tensor into a numpy array
    """
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
    """
    Add a dimension to an array
    """
    return arr.reshape(1, -1)


def pairs(iterable):
    """
    Returns all ordered pairs of items from an iterable
    """
    combos = list(combinations(iterable, 2))
    return combos + [(b, a) for a, b in combos]


class GracefulInterruptHandler(object):
    def __init__(self, sig=signal.SIGINT):
        self.sig = sig

    def __enter__(self):

        self.interrupted = False
        self.released = False

        self.original_handler = signal.getsignal(self.sig)

        def handler(signum, frame):
            self.release()
            self.interrupted = True

        signal.signal(self.sig, handler)

        return self

    def __exit__(self, type, value, tb):
        self.release()

    def release(self):

        if self.released:
            return False

        signal.signal(self.sig, self.original_handler)

        self.released = True

        return True
