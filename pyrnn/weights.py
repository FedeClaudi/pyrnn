import numpy as np


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
        if np.any(inhibitory):
            # otherwise dale ratio was not enforced
            self.weights[excitatory] = np.abs(self.weights[excitatory])
            self.weights[inhibitory] = -np.abs(self.weights[inhibitory])
