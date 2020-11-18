import matplotlib.pyplot as plt
from myterial import salmon
from sklearn.decomposition import PCA
from vedo.colors import colorMap
import networkx as nx
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from ._plot import clean_axes
from ._utils import npify, flatten_h


def plot_fixed_points_eigenvalues(fps, only_dominant=True):
    """
    Plots the eigenvalues of the jacobian an each
    fixed point in the complex plane.

    :param fps: list of FixedPoint objects
    :param only_dominant: bool, if true only the
        eigenvalue with largest magnitude for each FixedPoint
        is shown
    """
    f, ax = plt.subplots(figsize=(10, 10))

    # Plot unit circle
    t = np.linspace(0, 2 * np.pi, 360)
    ax.plot(np.cos(t), np.sin(t), lw=2, ls="--", color=[0.6, 0.6, 0.6])

    # Plot fixed points
    for fp in fps:
        evals = [emode.eigv for emode in fp.eigenmodes]
        mags = [np.abs(eval) for eval in evals]
        evals = np.array(evals)[np.argsort(mags)][::-1]

        colors = colorMap(
            np.array(mags)[np.argsort(mags)][::-1],
            name="bwr",
            vmin=0,
            vmax=1.5,
        )
        reals = np.real(evals)
        imgs = np.imag(evals)

        if only_dominant:
            ax.scatter(
                reals[0],
                imgs[0],
                color=colors[0],
                s=100,
                alpha=0.8,
                lw=1,
                edgecolors=[0.3, 0.3, 0.3],
            )
        else:
            ax.scatter(
                reals,
                imgs,
                c=colors,
                s=100,
                alpha=0.8,
                lw=1,
                edgecolors=[0.3, 0.3, 0.3],
            )

    # Clean up figure
    clean_axes(f)
    ax.spines["bottom"].set_position("zero")
    ax.spines["left"].set_position("zero")
    ax.axis("equal")
    ax.set(xticks=[-1.5, 1.5], yticks=[-1.5, 1.5])
    ax.set_xlabel("$\\Re$", fontsize=24, color=[0.3, 0.3, 0.3])
    ax.xaxis.set_label_coords(1, 0.48)
    ax.set_ylabel("$\\Im$", fontsize=24, color=[0.3, 0.3, 0.3])
    ax.yaxis.set_label_coords(0.4, 1)
    return f


def plot_training_loss(loss_history):
    """
    Simple plot with training loss trajectory

    Arguments:
        loss_history (list): loss at each epoch during training
    """
    f, ax = plt.subplots(figsize=(12, 7))

    ax.plot(loss_history, lw=2, color=salmon)
    ax.set(xlabel="epochs", ylabel="loss", title="Training loss")
    clean_axes(f)
    return f


def plot_recurrent_weights(model):
    """
    Plot a models recurrent weights as a heatmap

    Arguments:
        model (RNN): a built RNN
    """
    f, ax = plt.subplots(figsize=(10, 10))

    img = ax.imshow(npify(model.recurrent_weights, flatten=False), cmap="bwr")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    f.colorbar(img, cax=cax, orientation="vertical")

    ax.set(xticks=[], yticks=[], xlabel="units", ylabel="units")
    ax.axis("equal")
    clean_axes(f)
    return f


def plot_fps_graph(graph):
    """
    Plot a graph (nx.DiGraph) of fixed points connectivity

    Arguments:
        graph (nx.DiGraph): results of running FixedConnectivity analysis.
            A directed graph showing connections among fixed points
    """
    node_colors_lookup = {
        0: "lightseagreen",
        1: "lightsalmon",
        2: "powderblue",
        3: "thistle",
    }

    n_stable = [int(d["n_unstable"]) for n, d in graph.nodes(data=True)]
    nodes_colors = [node_colors_lookup.get(n, "salmon") for n in n_stable]

    pos = nx.spring_layout(graph)
    # nx.draw_networkx_edge_labels(graph, pos, edge_labels=None)
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_color=nodes_colors,
        node_size=200,
        edge_color="seagreen",
        edge_cmap=plt.cm.Greens,
    )
    plt.show()


def plot_render_state_history_pca_2d(
    hidden_history,
    lw=1,
    alpha=1,
    color="k",
):
    """
    Fits a PCA to high dim hidden state history
    and plots the result in 2d.

    Arguments:
        hidden_history (np.ndarray): array with history of hidden states
        lw (int): line weight of hidden state trace
        alpha(float): transparency of hidden state trace
        color (str): color of hidden state trace

    Returns:
        pca (PCA): PCA model fit to hidden history
    """
    hh = flatten_h(hidden_history)

    pca = PCA(n_components=3).fit(hh)

    pc = pca.transform(hh)

    f, ax = plt.subplots(figsize=(10, 10))
    ax.plot(pc[:, 0], pc[:, 1], lw=lw, color=color, alpha=alpha)
    clean_axes(f)
    return pca
