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

    def add_projection(
        self,
        from_region,
        to_region,
        probability,
        from_cell_type="all",
        to_cell_type="all",
    ):
        """
        Adds connections from one region to an other with some probability.

        Arguments:
            from_region: str. Name of Region from which connections come
            to_region: str. Name of Region receiving projections
            probability: float 0 <= probability <=1. Each unit in from_region
                connects to each unit in to_region with this probability
            from/to_cell_type: str ('all', 'excitatory', 'inhibitory'). Which cell type
                should the projections come from and target
        """

        def get_sign(cell_type):
            """converts a str with cell type
            to integer representation (1 for excitatory, 0 if 'all' and -1 for inhibitory)
            """
            return ["inhibitory", "all", "excitatory"].index(cell_type) - 1

        # get regions
        _from = self.regions[from_region]
        _to = self.regions[to_region]

        # select which array indices to modify
        rows = np.zeros(self.n_units)
        cols = np.zeros(self.n_units)
        rows[_to.idx : _to.end_idx] = 1
        cols[_from.idx : _from.end_idx] = 1

        # match by cell type
        if get_sign(from_cell_type) != 0:

            cols[self.sign != get_sign(from_cell_type)] = 0

        if get_sign(to_cell_type) != 0:
            rows[self.sign != get_sign(to_cell_type)] = 0

        # add connections
        logger.debug(
            f"Adding connections from {from_region} to {to_region} (p= {probability:.3f})"
        )
        self.connectivity = self.add_connections_with_probability(
            probability,
            self.connectivity,
            rows,
            cols,
        )

    @staticmethod
    def add_connections_with_probability(
        probability,
        arr,
        rows,
        cols,
    ):
        """
        Given a 2D numpy array it assigns to a portion of it an array of 1s choosen randomly with
        probability of probability (0s otherwise).

        Arguments:
            probability: float. Probability of connections
            arr: np.ndarray (2d array with connections matrix)
            rows, cols: np.ndarray (n_units x 1) with 1 for row/col to
                assign connections to and 0 elsewhere

        Returns:
            arr: modified array with connectivity
        """
        # create a lot of random connections
        random_connections = np.random.choice(
            [0, 1], size=arr.shape, p=[1 - probability, probability]
        )

        # prune unnecessary ones
        rows = np.array(1 - rows, dtype=bool)
        cols = np.array(1 - cols, dtype=bool)
        random_connections[rows, :] = 0
        random_connections[:, cols] = 0

        # add connections
        arr += random_connections
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
