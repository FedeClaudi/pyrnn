import matplotlib.pyplot as plt
from pyinspect._colors import salmon
import numpy as np
from vedo import Lines, show, Spheres, Sphere, Tube
from sklearn.decomposition import PCA
from rich import print


from ._plot import clean_axes, points_from_pc
from ._utils import prepend_dim


# -------------------------------- matplotlib -------------------------------- #


def plot_training_loss(loss_history):
    f, ax = plt.subplots(figsize=(12, 7))

    ax.plot(loss_history, lw=2, color=salmon)
    ax.set(xlabel="epochs", ylabel="loss", title="Training loss")
    clean_axes(f)


# ------------------------------- vedo renders ------------------------------- #
def render(actors, _show=True):
    for act in actors:
        act.lighting("off")

    if _show:
        print("[green]Render ready")
        show(*actors)


def plot_state_history_pca_3d(
    hidden_history, lw=30, alpha=0.1, color="k", pts=None, _show=True
):
    """
    Fits a PCA to high dim hidden state history
    and plots the result in 3d with vedo
    """

    pca = PCA(n_components=3).fit(hidden_history)

    pc = pca.transform(hidden_history)
    points = points_from_pc(pc)

    actors = [Lines(points).lw(lw).alpha(alpha).c(color)]

    if pts is not None:
        pts = pca.transform(pts)
        actors.append(Spheres(pts, r=0.15, c="r"))

    render(actors, _show=_show)
    return pca, actors


def plot_fixed_points(
    hidden_history, fixed_points, scale=0.5, _show=True, **kwargs
):
    pca, actors = plot_state_history_pca_3d(
        hidden_history, **kwargs, _show=False
    )

    t = pca.transform

    for fp in fixed_points:
        pos = t(fp.h.reshape(1, -1))[0, :]

        # Color based on the number of unstable modes
        if fp.n_unstable_modes == 0:
            color = "seagreen"
        elif fp.n_unstable_modes == 1:
            color = "salmon"
        elif fp.n_unstable_modes == 2:
            color = "skyblue"
        else:
            color = "magenta"

        # plot unstable directions
        for stable, eigval, eigvec in fp.eigenmodes:
            if not stable:
                delta = eigval * -eigvec * scale
                p0 = pos - t(np.real(delta).reshape(1, -1))
                p1 = pos + t(np.real(delta).reshape(1, -1))

                actors.append(
                    Tube([p1.ravel(), p0.ravel()], c=color, r=0.05, alpha=1)
                )

        actors.append(Sphere(pos, c=color, r=0.1))

    render(actors, _show=_show)
    return pca, actors


def plot_fixed_points_connectivity_analysis(
    hidden_history,
    fixed_points,
    fps_connectivity,
    scale=0.5,
    _show=True,
    **kwargs,
):
    pca, actors = plot_fixed_points(
        hidden_history, fixed_points, scale=0.5, _show=False, **kwargs
    )

    def t(arr):
        return pca.transform(prepend_dim(arr)).ravel()

    for start_point, end_point, trajectory in fps_connectivity:
        trajectory = [t(tp) for tp in trajectory]
        actors.append(Tube(trajectory, c="lightpink", r=0.02, alpha=1))
        actors.append(Sphere(trajectory[0], r=0.1, c="lightpink"))

    render(actors, _show=_show)
    return pca, actors
