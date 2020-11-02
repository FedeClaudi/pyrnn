import matplotlib.pyplot as plt
from pyinspect._colors import salmon
from vedo import Lines, show, Spheres, Sphere
from sklearn.decomposition import PCA


from ._plot import clean_axes, points_from_pc


def plot_training_loss(loss_history):
    f, ax = plt.subplots(figsize=(12, 7))

    ax.plot(loss_history, lw=2, color=salmon)
    ax.set(xlabel="epochs", ylabel="loss", title="Training loss")
    clean_axes(f)


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

    if _show:
        print("ready")
        show(*actors)

    return pca, actors


def plot_fixed_points(hidden_history, fixed_points, **kwargs):
    pca, actors = plot_state_history_pca_3d(
        hidden_history, **kwargs, _show=False
    )

    for fp in fixed_points:
        pos = pca.transform(fp.h.reshape(1, -1))[0, :]
        if fp.is_stable:
            color = "seagreen"
        else:
            color = "salmon"

        actors.append(Sphere(pos, c=color, r=0.15))

    print("ready")
    show(*actors)
