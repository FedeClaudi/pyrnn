import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from loguru import logger

from pyrnn._plot import create_triplot

"""
    Collection of functions/classes to:
        1. initialize recurrent weights in an RNN
        2. constrain the connectivity of an RNN e.g. to
            create a multi-regions RNN
"""


@dataclass
class Region:
    """
    Class representing a single region in
    a multi-region connectivity matrix
    """

    name: str
    n_units: str
    idx: int = 0  # specifies the index of the region's first unit in whole RNN's connectivity
    autopses: bool = True
    dale_ratio: float = None

    @property
    def connectivity(self):
        conn = np.ones((self.n_units, self.n_units))

        if not self.autopses:
            np.fill_diagonal(conn, 0)

        return conn

    @property
    def sign(self):
        if self.dale_ratio is None:
            return np.zeros(self.n_units)
        else:
            _sign = np.ones(self.n_units)
            n_excitatory = int(np.floor(len(_sign) * self.dale_ratio))
            _sign[n_excitatory:] = -1

            return _sign

    @property
    def end_idx(self):
        return self.idx + self.n_units


class MultiRegionConnectivity:
    def __init__(
        self,
        *regions,
    ):
        """
        Facilitates the creation of connectivity matrices (including inputs
        and outputs) for multi-region RNNs.

        Arugments:
            regions: variable number of Region objects. Ordering determines
                how the regions are ordered in the connectivity matrix.
        """
        # total number of units
        self.n_units = np.sum([r.n_units for r in regions])

        # create regions and adjust idx
        self.regions = {}
        idx = 0
        for region in regions:
            region.idx = idx
            self.regions[region.name] = region
            idx += region.n_units

        # create connectivity
        self.connectivity = np.zeros((self.n_units, self.n_units))
        for region in self.regions.values():
            self.connectivity[
                region.idx : region.end_idx, region.idx : region.end_idx
            ] = region.connectivity

        # create sign vector
        self.sign = np.hstack([r.sign for r in self.regions.values()])

        # create empty containers for input and output connectivity
        self.inputs = []
        self.outputs = []

    @property
    def W_in(self):
        """
        Input connectivity for the whole network
        """
        try:
            return np.vstack(self.inputs).T
        except ValueError:
            # no arrays
            return np.ones((self.n_units, 1))

    @property
    def W_rec(self):
        """
        Recurrent connectivity for the whole network
        """
        if self.sign is not None:
            return self.connectivity * self.sign
        else:
            return self.connectivity

    @property
    def W_out(self):
        """
        Output connectivity for the whole network
        """
        try:
            return np.vstack(self.outputs)
        except ValueError:
            # no arrays
            return np.ones((self.n_units, 1)).T

    @property
    def n_inputs(self):
        """
        Number of inputs to the network
        """
        return len(self.inputs)

    @property
    def n_outputs(self):
        """
        Network's outputs
        """
        return len(self.outputs)

    def add_input(self, *regions):
        """
        Add an input targeting specific regions.
        If no regions are passed the input targets all regions.

        Arguments:
            regions. variable number of string of regions names.
                Listed regions will recieve this input.
        """
        if not regions:
            inp = np.ones(self.n_units)
        else:
            inp = np.zeros(self.n_units)

            for region in regions:
                reg = self.regions[region]
                inp[reg.idx : reg.end_idx] = 1

        self.inputs.append(inp)

    def add_output(self, *regions):
        """
        Add an output from specific regions.
        If no regions are passed the outputis from all regions

        Arguments:
            regions. variable number of string of regions names.
                Listed regions will form the output.
        """
        if not regions:
            out = np.ones(self.n_units)
        else:
            out = np.zeros(self.n_units)

            for region in regions:
                reg = self.regions[region]
                out[reg.idx : reg.end_idx] = 1

        self.outputs.append(out)

    def add_projection(self, from_region, to_region, probability):
        """
        Adds connections from one region to an other with some probability.

        Arguments:
            from_region: str. Name of Region from which connections come
            to_region: str. Name of Region receiving projections
            probability: float 0 <= probability <=1. Each unit in from_region
                connects to each unit in to_region with this probability
        """
        # get regions
        _from = self.regions[from_region]
        _to = self.regions[to_region]

        # get indices of correct part of connectivity matrix
        if _from.idx < _to.idx:
            # feedforward projections
            row_start = _to.idx
            row_end = _to.end_idx
            col_start = _from.idx
            col_end = _from.end_idx
        else:
            # feedback projections
            row_start = _to.idx
            row_end = _to.end_idx
            col_start = _from.idx
            col_end = _from.end_idx

        # add connections
        logger.debug(
            f"Adding connections from {from_region} to {to_region} (p= {probability:.3f}) | row indices: ({row_start}-{row_end}) | col indices: ({col_start}-{col_end})"
        )
        self.connectivity = self.add_connections_with_probability(
            probability,
            self.connectivity,
            row_start,
            row_end,
            col_start,
            col_end,
        )

    @staticmethod
    def add_connections_with_probability(
        probability, arr, row_start, row_end, col_start, col_end
    ):
        """
        Given a 2D numpy array it assigns to a portion of it an array of 1s choosen randomly with
        probability of probability (0s otherwise).

        Arguments:
            probability: float. Probability of connections
            arr: np.ndarray (2d array with connections matrix)
            row_start, row_end, col_start, col_end: int. Indices used to select part of the array
                to which the random 0s-1s array is assigned

        Returns:
            arr: modified array with connectivity
        """

        shape = arr[row_start:row_end, col_start:col_end].shape
        arr[row_start:row_end, col_start:col_end] = np.random.choice(
            [0, 1], size=shape, p=[1 - probability, probability]
        )
        return arr

    def show(self):
        """
        Plot connectivity matrices
        """
        f, axes = create_triplot(figsize=(10, 10))

        axes.main.imshow(self.W_rec, cmap="bwr", vmin=-1, vmax=1)
        axes.top.imshow(self.W_in.T, cmap="bwr", vmin=-1, vmax=1)
        axes.right.imshow(self.W_out.T, cmap="bwr", vmin=-1, vmax=1)

        plt.show()
