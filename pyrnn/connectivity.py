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
    idx: int
    dale_ratio: float = None
    autopses: bool

    @property
    def connectivity(self):
        conn = np.ones((self.n_units, self.n_units))

        if not self.autopses:
            np.fill_diagonal(conn, 0)

        return conn

    @property
    def sign(self):
        if self.dale_ratio is None:
            return None
        else:
            _sign = np.ones(self.n_units)
            n_excitatory = int(np.floor(len(_sign) * self.dale_ratio))
            _sign[n_excitatory:] = -1

            return _sign

    @property
    def end_idx(self):
        return self.idx + self.n_units


class MultiRegionConnectivity:
    def __init__(self, autopses=True, dale_ratio=None, **regions):
        """
        Facilitates the creation of connectivity matrices (including inputs
        and outputs) for multi-region RNNs.

        Arugments:
            autopses: bool, True. Should autopses be included in connectivity?
            dale_ratio float, None. 0 < dale_ratio <1. Percentage of exitatory uinits.
                If dale_ratio is None units can be both excitatory and inhibitory.

            regions: name=n_units. Variable number of regions to add to RNN with their name
                and numberof units
        """
        # total number of units
        self.n_units = np.sum(list(regions.values()))

        # create regions
        self.regions = {}
        idx = 0
        for name, n_units in regions.items():
            self.regions[name] = Region(
                name, n_units, idx, dale_ratio, autopses
            )
            idx += n_units

        # create connectivity
        self.connectivity = np.zeros((self.n_units, self.n_units))
        for region in self.regions.values():
            self.connectivity[
                region.idx : region.end_idx, region.idx : region.end_idx
            ] = region.connectivity

        # create sign vector
        self.sign = (
            None
            if dale_ratio is None
            else np.hstack([r.sign for r in self.regions.values()])
        )

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
        return self.connectivity * self.sign

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
        size=connectivity[n_units[0] :, : n_units[0]].shape,
        p=[1 - prob_a_to_b, prob_a_to_b],
    )
    connectivity[n_units[0] :, : n_units[0]] = a_to_b

    b_to_a = np.random.choice(
        [0, 1],
        size=connectivity[: n_units[0], n_units[0] :].shape,
        p=[1 - prob_b_to_a, prob_b_to_a],
    )

    connectivity[: n_units[0], n_units[0] :] = b_to_a

    return connectivity


def define_connectivity(
    n_units=64,
    n_units_a=24,
    a_to_b_p=0.0,
    b_to_a_p=0,
    n_inputs=3,
    n_outputs=1,
):
    """
    Creates connectivity matrices for a two-regions RNN with inputs going
    only to the first region (a) and outputs coming out of only the second region (b)

    Arguments:
        n_units: int. Total numberof units
        n_units_a: int. Number of units in a
        a_to_b_p: float <1. Probability of connections a -> b
        b_to_a_p: float <1. Probability of connections b -> a
        n_inputs: int number of network inputs
        n_outputs: int number of network outputs

    Returns:
        connectivity matrices
    """
    n_units_b = n_units - n_units_a

    # create connectivity matrix for recurrent weights
    connectivity = constrained_connectivity(
        (n_units_a, n_units_b), a_to_b_p, b_to_a_p
    )

    # define input/output connectivity
    input_connectivity = np.vstack(
        [np.ones((n_units_a, n_inputs)), np.zeros((n_units_b, n_inputs))]
    )

    output_connectivity = np.vstack(
        [np.zeros((n_units_a, n_outputs)), np.ones((n_units_b, n_outputs))]
    ).T

    return connectivity, input_connectivity, output_connectivity


# ---------------------------- Weights initializer --------------------------- #


class RecurrentWeightsInitializer(object):
    """
    This class implements biological constraints
    on the recurrent weights of a RNN subclassing RNNBase
    """

    def __init__(
        self,
        initial_weights,
        dale_ratio=None,
        autopses=True,
        connectivity=None,
    ):
        """
        This class implements biological constraints
        on the recurrent weights of a RNN subclassing RNNBase.
        Updated weights can be accessed at RecurrentWeightsInitializer.weights.

        Arguments:
            initial_weights (np.ndarray): (n_units * n_units) np.array
                with recurrent weights initialized by pythorch.
            dale_ratio (float): of not None should be a float in range (0, 1)
                specifying which proportion of units should be excitatory
            autopses (bool): if False autopses are removed from weights
                (diagonal elements on weights matrix)
            connectivity: np.array of shape n_units x n_units with connectivity
                constraints for the recurrent layer
        """
        self.weights = initial_weights
        self.connectivity = np.ones_like(self.weights)

        if connectivity is not None:
            self._set_connectivity(connectivity)

        if dale_ratio is not None:
            self._apply_dale_ratio(dale_ratio)

        if not autopses:
            self._remove_autopses()

    def _remove_autopses(self):
        """
        Removes diagonal elements form weights matrix
        """
        np.fill_diagonal(self.weights, 0)
        return

    def _apply_dale_ratio(self, dale_ratio):
        """
        Implements the dale ratio to specify
        the proportion of excitatory/inhibitory units.

        Arguments:
            dale_ratio (float): in range (0, 1). Proportion of excitatory
                units
        """
        if dale_ratio < 0 or dale_ratio > 1:
            raise ValueError(f"Invalid dale ratio value of: {dale_ratio}")

        n_units = len(self.weights)
        n_excitatory = int(np.floor(n_units * dale_ratio))

        dale_vec = np.ones(n_units)
        dale_vec[n_excitatory:] = -1

        self.weights = np.matmul(np.abs(self.weights), np.diag(dale_vec))

    def _set_connectivity(self, connectivity):
        """
        Constraintes the RNN's recurrent weights connectivity with a given
        connectivity matrix (including sign)
        """
        if connectivity.shape != self.weights.shape:
            raise ValueError(
                "The connectivity constraint matrix should have shape (n_units x n_units)!"
            )

        # set weights to 0 where they need to be
        self.weights *= np.abs(connectivity)

        # enforce connectivity sign to match Dale ratio
        excitatory = np.sign(connectivity) == 1
        inhibitory = np.sign(connectivity) == -1
        self.weights[excitatory] = np.abs(self.weights[excitatory])
        self.weights[inhibitory] = -np.abs(self.weights[inhibitory])
