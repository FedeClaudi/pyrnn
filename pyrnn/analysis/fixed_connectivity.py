from scipy.spatial.distance import euclidean
import numpy as np
from rich import print
from rich.progress import track
from pyinspect._colors import mocassin, orange
from itertools import combinations_with_replacement as combinations
import networkx as nx

from pyrnn._utils import npify, torchify


class FixedPointsConnectivity(object):
    def __init__(
        self,
        model,
        fixed_points,
        n_initial_conditions=1024,
        noise_scale=0.1,
        dist_th=0.15,
    ):
        self.model = model
        self.fps = fixed_points
        self.n_initial_conditions = n_initial_conditions
        self.noise_scale = noise_scale
        self.dist_th = dist_th

    def _get_initial_conditions(self):
        inits_per_fp = int(self.n_initial_conditions / len(self.fps))

        n_units = self.fps[0].h.shape[0]
        inits = []
        for fp in self.fps:
            inits.extend(
                [
                    (fp, fp.h + np.random.normal(0, self.noise_scale, n_units))
                    for i in range(inits_per_fp)
                ]
            )

        inits = [(fp, torchify(ic)) for fp, ic in inits]
        return inits

    def _at_fp(self, fp):
        dists = [euclidean(f.h, fp) for f in self.fps]
        return np.min(dists) <= self.dist_th

    def _get_closest_fp(self, fp):
        dists = [euclidean(f.h, fp) for f in self.fps]
        closest = np.argmin(dists)
        return self.fps[closest]

    def _reconstruct_graph(self):
        se = [(o[0], o[1]) for o in self.outcomes]

        node_combinations = list(combinations(self.fps, 2))
        c1 = {(s, e): 0 for s, e in node_combinations}
        c2 = {(e, s): 0 for s, e in node_combinations}
        counts = {**c1, **c2}

        for start, end in se:
            counts[start, end] += 1

        counts = {c: v for c, v in counts.items() if v > 0}

        G = nx.Graph()

        for (s, e), w in counts.items():
            G.add_edge(s, e, weight=w)

        # TODO graph construction and drawing double check
        # a = 1

    def get_connectivity(self, const_input, max_iters=500):
        print(
            f"[{mocassin}]Extracting fixed points connectivity",
            f"([{orange}]{self.n_initial_conditions}[/{orange}] points)",
        )

        initial_conditions = self._get_initial_conditions()
        outcomes = []
        for start_fp, ic in track(
            initial_conditions, description="getting connectivity"
        ):
            h = ic
            trajectory = []

            for epoch in range(max_iters):
                o, h = self.model(const_input, h)

                _h = h.detach().numpy()
                trajectory.append(npify(_h))

                if self._at_fp(_h):
                    outcomes.append(
                        (start_fp, self._get_closest_fp(_h), trajectory)
                    )
                    break

        print(
            f"[{orange}]{len(outcomes)}[/{orange}][{mocassin}] initial conditions converged"
        )
        self.outcomes = outcomes

        self._reconstruct_graph()
        return outcomes
