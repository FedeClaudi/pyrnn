from scipy.spatial.distance import euclidean
import numpy as np
from rich import print
from rich.progress import track
from myterial import amber_light, orange

import networkx as nx

from pyrnn._utils import npify, torchify, pairs


class FixedPointsConnectivity(object):
    """
    Analyses the dynamics of a RNN in the absence
    of inputs to reconstruct the connectivity of
    the RNN's fixed points.
    This is done by starting the network in the proximity
    of each fixed point and observing at which fixed point
    the dynamics end up.

    Fixed points can be identified with pyrnn.analysis.fixed
    """

    def __init__(
        self,
        model,
        fixed_points,
        n_initial_conditions_per_fp=64,
        noise_scale=0.1,
        dist_th=0.15,
    ):
        """
        Args:
            model (RNNBase): instance of RNN class subclassing pyrnn._rnn.RNNBase
            fps (list): list of instances of pyrnn.analysis.fixed.FixedPoint
            n_initial_conditions_per_fp (int): number of initial conditions per fixed point
            noise_scale (float): std of the normal distributions used to sample noise for initial condition
            dist_th (float): when the network states get's within this distance to a FP the initial
                FP and that FP are counted as connected

        Returns:
            None

        """
        self.model = model
        self.fps = fixed_points
        self.inits_per_fp = n_initial_conditions_per_fp
        self.n_initial_conditions = n_initial_conditions_per_fp * len(
            fixed_points
        )
        self.noise_scale = noise_scale
        self.dist_th = dist_th

    def _get_initial_conditions(self):
        """
        Get's a list of points (i.e. hidden states) around
        each given fixed point by adding some normally distributed
        noise to each FP's state.

        Returns:
            initial conditons (list): list of tuples of initial conditions.
                Each touple has the original fixed point and the noisy
                state of the initial condition.
        """
        n_units = self.fps[0].h.shape[0]

        # Get states with noise
        inits = []
        for fp in self.fps:
            inits.extend(
                [
                    (fp, fp.h + np.random.normal(0, self.noise_scale, n_units))
                    for i in range(self.inits_per_fp)
                ]
            )

        # Turn states into tensors
        inits = [(fp, torchify(ic)) for fp, ic in inits]
        return inits

    def _at_fp(self, point):
        """
        Check if the current point (h) is within the
        distance threshold from any fixed point

        Arguments:
            point (np.ndarray): RNN's hidden state

        Returns:
            bool: True if the current point is close to a FP
        """
        dists = [euclidean(f.h, point) for f in self.fps]
        return np.min(dists) <= self.dist_th

    def _get_closest_fp(self, point):
        """
        Get's the fixed point that is closes to the current
        point.

        Arguments:
            point (np.ndarray): RNN's hidden state

        Returns:
            FixedPoint: closest fixed point

        """
        dists = [euclidean(f.h, point) for f in self.fps]
        closest = np.argmin(dists)
        return self.fps[closest]

    def _reconstruct_graph(self, connections):
        """
        Given a dictionary showing how many times
        initial conditions from a given FP ended up
        in any other FP (for all FPS), reconstructs a
        graph showing the FPS connectivity.

        Argument:
            connections (dict): keys are tuples (FP1, FP2) and
                values indicate how many times startin at FP1
                you end up at FP2

        Returns:
            nx.DiGraph: directed graph showing connections
                between FPS
        """
        connections = {k: v for k, v in connections.items() if v > 0}

        graph = nx.DiGraph()
        for (fp1, fp2), w in connections.items():
            graph.add_node(fp1.fp_id, n_unstable=fp1.n_unstable_modes, fp=fp1)
            graph.add_node(fp2.fp_id, n_unstable=fp2.n_unstable_modes, fp=fp2)

            graph.add_edge(
                fp1.fp_id,
                fp2.fp_id,
                weight=w,
                fp1=fp1,
                fp2=fp2,
                prob=w / self.inits_per_fp,
            )
        return graph

    def get_connectivity(self, const_input, max_iters=500):
        """
        Reconstructs the connectivity between fixed points.
        For each initial condition let the dynamics evolve under
        constant input and see at which FP you end up.
        Keep track of this for all initial conditions and all FPS,
        then use this information to reconstruct the connectivity of FPS.

        Arguments:
            const_input (np.ndarray): properly shaped 3d np.array with
                constant inputs (generally 0s). Shape should be:
                (1, 1, n_inputs).
            max_iters (int): for each initial condition, run max
                max_iters steps. If state didn't converge to a FP by then
                stop.

        Returns:
            nx.DiGraph: directed graph showing connections
                between FPS
        """
        print(
            f"[{amber_light}]Extracting fixed points connectivity (",
            f"[{orange}]{self.n_initial_conditions}[/{orange}][{amber_light}] points)",
        )

        # Get initial conditions
        initial_conditions = self._get_initial_conditions()

        outcomes = []
        connections = {p: 0 for p in pairs(self.fps)}  # keys are (FP1, FP2)

        # loop over initial conditions
        for start_fp, ic in track(
            initial_conditions, description=f"[{orange}]getting connectivity"
        ):
            h = ic.reshape(1, 1, -1)
            trajectory = []  # keep track of h's trajectory

            # loop over epochs
            for epoch in range(max_iters):
                # step network
                o, h = self.model(const_input, h)

                # get current state
                _h = h.detach().numpy()
                trajectory.append(npify(_h))

                # check if we reached a FP
                if self._at_fp(_h):
                    end_fp = self._get_closest_fp(_h)
                    outcomes.append((start_fp, end_fp, trajectory))
                    connections[(start_fp, end_fp)] += 1
                    break

        print(
            f"[{orange}]{len(outcomes)}[/{orange}][{amber_light}] initial conditions converged"
        )
        self.outcomes = outcomes

        # Reconstruct graph
        graph = self._reconstruct_graph(connections)
        return outcomes, graph
