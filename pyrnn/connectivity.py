import numpy as np


"""
    Collection of functions/classes to:
        1. initialize recurrent weights in an RNN
        2. constrain the connectivity of an RNN e.g. to
            create a multi-regions RNN
"""


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
        connectivity matrix
        """
        if connectivity.shape != self.weights.shape:
            raise ValueError(
                "The connectivity constraint matrix should have shape (n_units x n_units)!"
            )

        self.weights *= connectivity
