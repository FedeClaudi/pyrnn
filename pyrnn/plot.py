import matplotlib.pyplot as plt
from pyinspect._colors import salmon
from vedo import Lines, show, Spheres
from sklearn.decomposition import PCA


from ._plot import clean_axes


def plot_training_loss(loss_history):
    f, ax = plt.subplots(figsize=(12, 7))

    ax.plot(loss_history, lw=2, color=salmon)
    ax.set(xlabel="epochs", ylabel="loss", title="Training loss")
    clean_axes(f)


def plot_state_history_pca_3d(
    hidden_history, lw=30, alpha=0.1, color="k", pts=None
):
    """
    Fits a PCA to high dim hidden state history
    and plots the result in 3d with vedo
    """

    pca = PCA(n_components=3).fit(hidden_history)

    pc = pca.transform(hidden_history)
    points = [[pc[i, :], pc[i + 1, :]] for i in range(len(pc) - 1)]

    actors = [Lines(points).lw(lw).alpha(alpha).c(color)]

    if pts is not None:
        pts = pca.transform(pts)
        actors.append(Spheres(pts, r=0.15, c="r"))

    print("ready")
    show(*actors)

    return pca
