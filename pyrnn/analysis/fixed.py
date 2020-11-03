import numpy as np
import torch
from rich import print
from pyinspect._colors import mocassin, orange
from scipy.spatial.distance import euclidean
from collections import namedtuple

from pyrnn._progress import fixed_points_progress
from pyrnn._io import save_json, load_json
from pyrnn._utils import flatten_h

eig_mode = namedtuple("eigmode", "stable, eigv, eigvec")


class FixedPoint(object):
    def __init__(self, h, constant_input, model=None, jacobian=None):
        self.constant_input = constant_input
        self.h = h
        self.model = model

        if jacobian is None:
            self.compute_jacobian()
            self.analyse_stability()
        else:
            self.jacobian = jacobian

    def to_dict(self):
        return dict(
            h=self.h.tolist(),
            constant_input=self.constant_input.tolist(),
            jacobian=self.jacobian.tolist(),
        )

    @classmethod
    def from_dict(cls, data_dict):
        h = np.array(data_dict["h"])
        constant_input = np.array(data_dict["constant_input"])
        jacobian = np.array(data_dict["jacobian"])
        fp = cls(h, constant_input, jacobian=jacobian)
        fp.analyse_stability()
        return fp

    def compute_jacobian(self):
        n_units = len(self.h)
        jacobian = torch.zeros(n_units, n_units)

        h = torch.from_numpy(self.h.astype(np.float32)).reshape(1, 1, -1)
        h.requires_grad = True
        _o, _h = self.model(self.constant_input, h)

        for i in range(n_units):
            output = torch.zeros(1, 1, n_units)
            output[0, 0, i] = 1

            g = torch.autograd.grad(
                _h, h, grad_outputs=output, retain_graph=True
            )[0]
            jacobian[:, i : i + 1] = g[0, 0, :].reshape(-1, 1)

        self.jacobian = jacobian.numpy()

    def decompose_jacobian(self):
        # return eigen values and eigen vectors of jacobain
        return np.linalg.eig(self.jacobian)

    def analyse_stability(self):
        eigv, eigvecs = self.decompose_jacobian()

        # Get overall stability
        self.is_stable = np.all(np.abs(eigv) < 1.0)

        # Get stability over each mode
        self.eigenmodes = []  # holds stable eigenvecs
        for e_val, e_vec in zip(eigv, eigvecs.T):
            # Magnitude of complex eigenvalue
            eigv_mag = np.abs(e_val)

            if eigv_mag <= 1.0:
                stable = True
            else:
                stable = False

            self.eigenmodes.append(eig_mode(stable, eigv_mag, np.real(e_vec)))

        self.n_unstable_modes = np.sum(
            [1 for mode in self.eigenmodes if not mode.stable]
        )


class FixedPoints(object):
    def __init__(
        self,
        model,
        speed_tol=1e-05,
        gamma=0.01,
        dist_th=0.15,
        noise_scale=None,
    ):
        self.speed_tol = speed_tol
        self.gamma = gamma
        self.dist_th = dist_th
        self.noise_scale = noise_scale or 0.0

        self.model = model

    def _get_initial_conditions(self, hidden, n_initial_conditions):
        random_times = np.random.randint(0, len(hidden), n_initial_conditions)
        initial_conditions = [hidden[s, :] for s in random_times]

        if self.noise_scale:
            n = initial_conditions[0].shape[0]
            initial_conditions = [
                ic + np.random.normal(0, scale=self.noise_scale, size=n)
                for ic in initial_conditions
            ]

        return initial_conditions

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
        lr_decay_epoch=500,
        max_fixed_points=100,
    ):
        # Flatten hidden
        hidden = flatten_h(hidden)

        print(f"[{mocassin}]Looking for fixed points.")
        initial_conditions = self._get_initial_conditions(
            hidden, n_initial_conditions
        )

        fixed_points = []
        with fixed_points_progress as progress:
            main_tid = progress.add_task(
                    f"[bold {orange}] Finding fixed points",
                    start=True,
                    total=n_initial_conditions,
                    fpspeed=None,
                )

            # loop over initial conditions
            for nhid, hid in enumerate(initial_conditions):
                progress.update(
                    main_tid, completed=nhid, fpspeed=None,
                )

                # Add a second progress bar for each initial conditon
                tid = progress.add_task(
                    f"[{mocassin}] Init.cond.: {nhid}/{n_initial_conditions} | ({len(fixed_points)})",
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
        if fixed_points:
            self.fixed_points = [
                FixedPoint(fp, constant_inputs[0], self.model)
                for fp in fixed_points
            ]
            return self.fixed_points
        else:
            self.fixed_points = []
            return None

    def save_fixed_points(self, filepath):
        save_json(filepath, [fp.to_dict() for fp in self.fixed_points])

    @staticmethod
    def load_fixed_points(filepath):
        data = load_json(filepath)
        return [FixedPoint.from_dict(d) for d in data]
