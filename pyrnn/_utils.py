import numpy as np
import torch
from itertools import combinations_with_replacement as combinations
import signal
from einops import repeat


def constrained_connectivity(n_units, prob_a_to_b, prob_b_to_a):
    """
    Creates a n_units x n_units array with constrained recurrent connectivity
    between two regions to initialize recurrent weights

    Arguments:
        n_units: tuple of two int. Number of units in first and second subregion
        prob_a_to_b: 0 <= float <= 1. Probability of a connection from first
            to second subregion
        prob_b_to_a: 0 <= float <= 1. Probability of a connection from second
            to first subregion
    """
    # initialize
    tot_units = n_units[0] + n_units[1]
    connectivity = np.zeros((tot_units, tot_units))

    # define sub regions
    connectivity[: n_units[0], : n_units[0]] = 1
    connectivity[n_units[0] :, n_units[0] :] = 1

    # define inter region connections
    a_to_b = np.random.choice(
        [0, 1],
        size=connectivity[: n_units[0], n_units[0] :].shape,
        p=[1 - prob_a_to_b, prob_a_to_b],
    )
    connectivity[: n_units[0], n_units[0] :] = a_to_b

    b_to_a = np.random.choice(
        [0, 1],
        size=connectivity[n_units[0] :, : n_units[0]].shape,
        p=[1 - prob_b_to_a, prob_b_to_a],
    )
    connectivity[n_units[0] :, : n_units[0]] = b_to_a

    return connectivity


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
        try:
            tensor = tensor.detach().numpy()
        except TypeError:
            tensor = tensor.detach().cpu().numpy()

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
    return repeat(arr, "i -> n i", n=1)


def pairs(iterable):
    """
    Returns all ordered pairs of items from an iterable
    """
    combos = list(combinations(iterable, 2))
    return combos + [(b, a) for a, b in combos]


class GracefulInterruptHandler(object):
    def __init__(self, sig=signal.SIGINT):
        """
        Used as a context captures CTRL+C
        events and handles them gracefully.
        """
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
