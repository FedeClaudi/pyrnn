from scipy.spatial.distance import euclidean
import numpy as np
from rich import print
from rich.progress import track
from pyinspect._colors import mocassin, orange

import networkx as nx

from pyrnn._utils import npify, torchify, pairs


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
        return inits, inits_per_fp

    def _at_fp(self, fp):
        dists = [euclidean(f.h, fp) for f in self.fps]
        return np.min(dists) <= self.dist_th

    def _get_closest_fp(self, fp):
        dists = [euclidean(f.h, fp) for f in self.fps]
        closest = np.argmin(dists)
        return self.fps[closest]

    def _reconstruct_graph(self, connections, inits_per_fp):
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
                prob=w / inits_per_fp,
            )
        return graph

    def get_connectivity(self, const_input, max_iters=500):
        print(
            f"[{mocassin}]Extracting fixed points connectivity (",
            f"[{orange}]{self.n_initial_conditions}[/{orange}][{mocassin}] points)",
        )

        initial_conditions, inits_per_fp = self._get_initial_conditions()
        outcomes = []
        connections = {p: 0 for p in pairs(self.fps)}

        for start_fp, ic in track(
            initial_conditions, description=f"[{orange}]getting connectivity"
        ):
            h = ic.reshape(1, 1, -1)
            trajectory = []

            for epoch in range(max_iters):
                o, h = self.model(const_input, h)

                _h = h.detach().numpy()
                trajectory.append(npify(_h))

                if self._at_fp(_h):
                    end_fp = self._get_closest_fp(_h)
                    outcomes.append((start_fp, end_fp, trajectory))
                    connections[(start_fp, end_fp)] += 1
                    break

        print(
            f"[{orange}]{len(outcomes)}[/{orange}][{mocassin}] initial conditions converged"
        )
        self.outcomes = outcomes

        graph = self._reconstruct_graph(connections, inits_per_fp)
        return outcomes, graph
