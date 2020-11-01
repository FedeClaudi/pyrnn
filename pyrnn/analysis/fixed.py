import numpy as np
import torch
from rich import print
from pyinspect._colors import mocassin, orange
from scipy.spatial.distance import euclidean

from pyrnn._progress import fixed_points_progress


class FixedPoints(object):
    def __init__(self, model, speed_tol=1e-05, gamma=0.01, dist_th=0.15):
        self.speed_tol = speed_tol
        self.gamma = gamma
        self.dist_th = dist_th

        self.model = model

    def _get_initial_conditions(self, hidden, n_initial_conditions):
        random_times = np.random.randint(0, len(hidden), n_initial_conditions)
        return [hidden[s, :] for s in random_times]

    def _append_fixed_point(self, fps, fp):
        if fps:
            dists = [euclidean(f, fp) for f in fps]
            if np.min(dists) < self.dist_th:
                return fps
        return fps + [fp]

    def _run_initial_condition(
        self, hid, constant_inputs, progress, tid, max_iters, lr_decay_epoch
    ):
        # loop over inputs
        for n_cn, constant_input in enumerate(constant_inputs):
            gamma = self.gamma

            h = torch.from_numpy(hid.astype(np.float32)).reshape(1, 1, -1)
            h.requires_grad = True
            h.retain_grad()

            for epoch in range(max_iters):
                # step RNN
                _, _h = self.model(constant_input, h)

                # Compute
                q = torch.norm(h - _h)

                # Step
                if q < self.speed_tol:
                    return h.detach().numpy().ravel()
                else:
                    q.backward()
                    if epoch % lr_decay_epoch == 0 and epoch > 0:
                        gamma *= 0.5

                    h = h - gamma * h.grad
                    h.retain_grad()

                progress.update(
                    tid, completed=epoch * (n_cn + 1), fpspeed=q.item()
                )
        return None

    def find_fixed_points(
        self,
        hidden,
        constant_inputs,
        n_initial_conditions=100,
        max_iters=500,
        lr_decay_epoch=100,
        max_fixed_points=100,
    ):
        print(f"[{mocassin}]Looking for fixed points.")
        initial_conditions = self._get_initial_conditions(
            hidden, n_initial_conditions
        )

        fixed_points = []
        with fixed_points_progress as progress:
            # loop over initial conditions
            for nhid, hid in enumerate(initial_conditions):
                tid = progress.add_task(
                    f"[bold green] Init.cond.: {nhid}/{n_initial_conditions} | found: {len(fixed_points)}",
                    start=True,
                    total=max_iters * len(constant_inputs),
                    fpspeed=None,
                )

                fp = self._run_initial_condition(
                    hid,
                    constant_inputs,
                    progress,
                    tid,
                    max_iters,
                    lr_decay_epoch,
                )
                if fp is not None:
                    fixed_points = self._append_fixed_point(fixed_points, fp)

                progress.remove_task(tid)
                if len(fixed_points) >= max_fixed_points:
                    break

        print(
            f"[{mocassin}]Found [{orange}]{len(fixed_points)}[/{orange}] from [{orange}]{n_initial_conditions}[/{orange}] initial conditions"
        )
        return np.vstack(fixed_points)  # n_fps x n_units
