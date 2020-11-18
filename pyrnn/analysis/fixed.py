import numpy as np
import torch
from rich import print
from myterial import amber_light, orange, cyan
from scipy.spatial.distance import euclidean
from collections import namedtuple
from pyinspect import Report

from pyrnn._progress import fixed_points_progress
from pyrnn._io import save_json, load_json
from pyrnn._utils import flatten_h, GracefulInterruptHandler, torchify

# named tuple storing eigen modes info
eig_mode = namedtuple("eigmode", "stable, eigv, eigvec")


def list_fixed_points(fps):
    """
    Prints an overview of a list of fixed points

    Argument:
        fps: list of FixedPoint objects
    """

    rep = Report(title="fps", color=amber_light, accent=orange)
    fps = sorted(fps, key=lambda fp: fp.n_unstable_modes)
    for fp in fps:
        s = f"[b {orange}]{fp.fp_id:03}[/b {orange}] - Stable: [{cyan}]{fp.is_stable}[/{cyan}]"
        if not fp.is_stable:
            s += f" | n unstable modes: {fp.n_unstable_modes}"
        rep.add(s)
    rep.print()


class FixedPoint(object):
    """
    Class representing a single Fixed Point
    """

    def __init__(self, fp_id, h, constant_input, model=None, jacobian=None):
        """
        A single fixed point and the corresponding hidden state.
        Can be used to compute the jacobian of the dynamics at the fixed point and
        used the jacobian to find stable/unstable modes.

        Arguments:
            fp_id (int): id of the fixed point
            h (np.ndarray): hidden state of the network at fixed point
            constant_input (np.ndarray): constant input used to find fixed point
            model (RNN): instance of a RNN class
            jacobian (np.ndarray): jacobian of the dynamics at the fixed point.
                If None, the jacobian is computed (requires model to not be None)
        """
        self.constant_input = constant_input
        self.h = h
        self.model = model
        self.fp_id = fp_id

        if jacobian is None:
            self.compute_jacobian()
            self.analyse_stability()
        else:
            self.jacobian = jacobian

    def __repr__(self):
        return f"FixedPoint ({self.fp_id})"

    def __str__(self):
        return f"FixedPoint ({self.fp_id})"

    def to_dict(self):
        """
        Returns the fixed point's attributs
        as a dictionary, used to save FPS to
        a .json file
        """
        return dict(
            fp_id=self.fp_id,
            h=self.h.tolist(),
            constant_input=self.constant_input.tolist(),
            jacobian=self.jacobian.tolist(),
        )

    @classmethod
    def from_dict(cls, fp_id, data_dict):
        """
        Creates an instance of FP from a dictionary
        of attributes, used when loading FPS from
        a .json file.
        """
        h = np.array(data_dict["h"])
        constant_input = np.array(data_dict["constant_input"])
        jacobian = np.array(data_dict["jacobian"])
        fp = cls(fp_id, h, constant_input, jacobian=jacobian)
        fp.analyse_stability()
        return fp

    def compute_jacobian(self):
        """
        Computes the jacobian of the dynamics at
        the fixed point's hidden state.
        """
        n_units = len(self.h)
        jacobian = torch.zeros(n_units, n_units)

        # initialize hidden state
        h = torchify(self.h)
        h.requires_grad = True

        _o, _h = self.model(self.constant_input, h)
        # Loop over each dimension of the hidden state vector
        for i in range(n_units):
            output = torch.zeros(1, 1, n_units)
            output[0, 0, i] = 1

            g = torch.autograd.grad(
                _h, h, grad_outputs=output, retain_graph=True
            )[0]
            jacobian[:, i : i + 1] = g[0, 0, :].reshape(-1, 1)

        self.jacobian = jacobian.numpy()

    def decompose_jacobian(self):
        """
        return eigen values and eigen vectors of jacobain
        """
        return np.linalg.eig(self.jacobian)

    def analyse_stability(self):
        """
        Inspects the magnitude of the eigen values
        of the dynamic's Jacobian to detect
        stable/unstable modes
        """
        eigv, eigvecs = self.decompose_jacobian()

        # Get overall stability (all modes stable)
        self.is_stable = np.all(np.abs(eigv) < 1.0)

        # Sort by eigv
        sort_idx = np.argsort(eigv)
        eigv = eigv[sort_idx]
        eigvecs = eigvecs[:, sort_idx]

        # Get stability over each mode
        self.eigenmodes = []  # holds stable eigenvecs
        for e_val, e_vec in zip(eigv, eigvecs.T):
            # Magnitude of complex eigenvalue
            eigv_mag = np.abs(e_val)

            if eigv_mag <= 1.0:
                stable = True
            else:
                stable = False

            # store the stability, eigenvalue and eigenvectors
            self.eigenmodes.append(eig_mode(stable, e_val, np.real(e_vec)))

        # count number of untable modes
        self.n_unstable_modes = np.sum(
            [1 for mode in self.eigenmodes if not mode.stable]
        ).astype(np.int32)


class FixedPoints(object):
    """
    Analyze a RNN's dynamics under constant inputs
    to find fixed points.
    Inspired by: https://www.mitpressjournals.org/doi/full/10.1162/NECO_a_00409
    "Opening the Black Box: Low-Dimensional Dynamics in High-Dimensional Recurrent Neural Networks"
    Sussillo and Barak 2013.
    """

    def __init__(
        self,
        model,
        speed_tol=1e-05,
        dist_th=0.15,
        noise_scale=None,
    ):
        """
        Analyze a RNN's dynamics under constant inputs
        to find fixed points.

        Arguments:
            model (RNN): instance of an RNN class
            speed_tol (float): when the dynamic's speed are blow this threshold
                the state is considered to be a fixed point
            dist_th (float): if a found FP is within this distance from another FP,
                they're considered to be the same FP (to avoid duplications)
            noise_scale (float): std of the normal distribution used to inject noise
                in the initial conditions
        """
        self.speed_tol = speed_tol
        self.dist_th = dist_th
        self.noise_scale = noise_scale or 0.0

        self.model = model

    def __repr__(self):
        return f"FixedPoints (# {self.fp_id} fps)"

    def __str__(self):
        return f"FixedPoints (# {self.fp_id} fps)"

    def _get_initial_conditions(self, hidden, n_initial_conditions):
        """
        Get set of initial conditions for the analysis.
        They're computed by taking random points along a trajectory
        of hidden states and adding some noise.

        Arguments:
            hidden (np.ndarray): trajectory of hidden states
            n_initial_conditions (int): number of initial conditions

        Returns:
            initial_conditions (list): list of np.arrays with hidden
                state for each initial condition
        """
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
        """
        If a FP is far enough from the FPS found so far,
        keep it.

        Arguments:
            fps (list): list of FPS
            fp (np.array): hidden state of the currently considered fp.

        Returns
            fps (list): list of FPS
        """
        if fps:
            dists = [euclidean(f, fp) for f in fps]
            if np.min(dists) < self.dist_th:
                return fps
        return fps + [fp]

    def _run_initial_condition(
        self, hid, constant_inputs, progress, tid, max_iters, lr_decay_epoch
    ):
        """
        Starting the network at an initial condition keeps stepping the RNN
        with constant input. Then computes the dynamic's velocity and if
        it's small enough it consider's the fixed point to be found.
        Every N steps the width of the steps taken is decreased.

        Arguments:
            hid (np.array): hidden state of initial condition
            constant_inputs (list): list of np.arrays with constant inputs
            progress (Progress): progress bar context manager
            tid (id): id of the progress bar task
            max_iters (int): max iterations that each initial condition
                is run for
            lr_decay_epoch (int): every lr_decay_epoch iterations the
                step width is reduced by a factor of gamma
        """
        # loop over inputs
        with GracefulInterruptHandler() as handler:
            for n_cn, constant_input in enumerate(constant_inputs):
                gamma = self.gamma

                h = torch.from_numpy(hid.astype(np.float32)).reshape(1, 1, -1)
                h.requires_grad = True
                h.retain_grad()

                # loop over iterations
                for epoch in range(max_iters):
                    # step RNN
                    _, _h = self.model(constant_input, h)

                    # Compute
                    q = torch.norm(h - _h)

                    # Step
                    if q < self.speed_tol:
                        # found a FP
                        return h.detach().numpy().ravel()
                    else:
                        # step in the direction of decreasing speed
                        q.backward()
                        if epoch % lr_decay_epoch == 0 and epoch > 0:
                            gamma *= 0.5

                        # update state
                        h = h - gamma * h.grad
                        h.retain_grad()

                    if handler.interrupted:
                        return False

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
        gamma=0.01,
    ):
        """
        Runs analysis to find fixed points. For N initial conditions
        let the dynamics evolve under constant input and observe if
        they reach a point where they are slow enough.

        Arguments:
            hidden (np.array): hidden state of initial condition
            constant_inputs (list): list of np.arrays with constant inputs
            n_initial_conditions (int): number of initial conditions to consider
            max_iters (int): max iterations that each initial condition
                is run for
            lr_decay_epoch (int): every lr_decay_epoch iterations the
                step width is reduced by a factor of gamma
            gamma (float)L factor by which step size is reduced
            max_fixed_points (int): when this number of fixed points is found
                the analysis is stopped even though not all initial conditions
                might have been used so far
        """
        self.gamma = gamma

        # Flatten hidden
        hidden = flatten_h(hidden)

        print(f"[{amber_light}]Looking for fixed points.")
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
            with GracefulInterruptHandler() as h:
                for nhid, hid in enumerate(initial_conditions):
                    progress.update(
                        main_tid,
                        completed=nhid,
                        fpspeed=None,
                    )

                    # Add a second progress bar for each initial conditon
                    tid = progress.add_task(
                        f"[{amber_light}] Init.cond.: {nhid}/{n_initial_conditions} | ({len(fixed_points)}/{max_fixed_points})",
                        start=True,
                        total=max_iters * len(constant_inputs),
                        fpspeed=None,
                    )

                    # Run initial condition to find a FP
                    fp = self._run_initial_condition(
                        hid,
                        constant_inputs,
                        progress,
                        tid,
                        max_iters,
                        lr_decay_epoch,
                    )
                    if fp is False or h.interrupted:
                        break

                    if fp is not None:
                        fixed_points = self._append_fixed_point(
                            fixed_points, fp
                        )

                    progress.remove_task(tid)
                    if len(fixed_points) >= max_fixed_points:
                        break

        # Create instance of FixedPoint for each fixed point state found so far
        print(
            f"[{amber_light}]Found [{orange}]{len(fixed_points)}[/{orange}] from [{orange}]{n_initial_conditions}[/{orange}] initial conditions"
        )
        if fixed_points:
            self.fixed_points = [
                FixedPoint(n, fp, constant_inputs[0], self.model)
                for n, fp in enumerate(fixed_points)
            ]
            return self.fixed_points
        else:
            self.fixed_points = []
            return None

    def save_fixed_points(self, filepath):
        """
        Saves the fixed points found to a .json file
        """
        print(f"[{amber_light}]Saving fixed points at: [{orange}]{filepath}")
        save_json(filepath, [fp.to_dict() for fp in self.fixed_points])

    @staticmethod
    def load_fixed_points(filepath):
        """
        Load fixed points from a .json file
        """
        print(
            f"[{amber_light}]Loading fixed points from: [{orange}]{filepath}"
        )
        data = load_json(filepath)
        return [FixedPoint.from_dict(n, d) for n, d in enumerate(data)]
