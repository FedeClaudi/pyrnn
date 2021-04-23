import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from vedo.colors import colorMap
import networkx as nx
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from myterial import salmon

from pyrnn._plot import (
    clean_axes,
    calc_nrows_ncols,
    center_axes,
    create_triplot,
)
from pyrnn._utils import npify, flatten_h
from pyrnn.linalg import classify_equilibrium


def plot_eigenvalues_magnitudes(evals, ax=None, color=None, alpha=None):
    """
    Plot the magnitude of a list of eigenvalues

    Arguments:
        evals: list of np.array of eigenvalues
    """
    color = color or [0.3, 0.3, 0.3]
    alpha = alpha or 1
    if ax is None:
        f, ax = plt.subplots(figsize=(12, 8))
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    mags = sorted([np.abs(eval) for eval in evals], reverse=True)

    ax.axhline(1, lw=3, ls="--", color=[0.3, 0.3, 0.3], alpha=0.5)
    ax.plot(mags, "-", lw=3, color=[0.3, 0.3, 0.3])
    ax.scatter(
        np.arange(len(mags)),
        mags,
        color="w",
        edgecolors=color,
        alpha=alpha,
        lw=2,
        zorder=100,
    )

    ax.set(
        ylabel="$|\\lambda|$",
        xlabel="Eigenvalue index",
        title=classify_equilibrium(evals),
    )
    return ax


def plot_eigenvalues(evals, only_dominant=False, ax=None):
    """
    Plot a list of eigenvalues in the complex plane

    Arguments:
        evals: list of np.array of eigenvalues
        ax: axes object, optional.
        only_dominant: bool, if True only the largest eval is shown
    """
    if ax is None:
        f, ax = plt.subplots(figsize=(10, 10))

        # Plot unit circle
        t = np.linspace(0, 2 * np.pi, 360)
        ax.plot(np.cos(t), np.sin(t), lw=2, ls="--", color=[0.6, 0.6, 0.6])

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
    center_axes(ax)

    ax.axis("equal")
    ax.set(xticks=[-1.5, 1.5], yticks=[-1.5, 1.5])
    ax.set_xlabel("$\\Re$", fontsize=12, color=[0.3, 0.3, 0.3])
    ax.xaxis.set_label_coords(1, 0.48)
    ax.set_ylabel("$\\Im$", fontsize=12, color=[0.3, 0.3, 0.3])
    ax.yaxis.set_label_coords(0.4, 1)

    return ax


def plot_fixed_points_eigenvalues(fps, only_dominant=True):
    """
    Plots the eigenvalues of the jacobian an each
    fixed point in the complex plane.

    :param fps: list of FixedPoint objects
    :param only_dominant: bool, if true only the
        eigenvalue with largest magnitude for each FixedPoint
        is shown
    """
    nrows, ncols = calc_nrows_ncols(len(fps), aspect=(12, 12))
    f, axarr = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(12, 12), sharex=True
    )
    axarr = axarr.flatten()

    f2, ax2 = plt.subplots(figsize=(12, 8))

    # for unit circle
    t = np.linspace(0, 2 * np.pi, 360)

    # Plot fixed points
    for n, (ax, fp) in enumerate(zip(axarr, fps)):
        evals = [emode.eigv for emode in fp.eigenmodes]
        plot_eigenvalues(evals, only_dominant=only_dominant, ax=ax)

        # plot unit circle
        ax.plot(np.cos(t), np.sin(t), lw=2, ls="--", color=[0.6, 0.6, 0.6])

        # Plot eigenvalues magnitued
        col = colorMap(fp.n_unstable_modes, name="viridis", vmin=0, vmax=4)
        plot_eigenvalues_magnitudes(evals, ax=ax2, color=col)

    for ax in axarr[len(fps) :]:
        ax.axis("off")

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


def plot_recurrent_weights(model, ax=None, scalebar=True):
    """
    Plot a models recurrent weights as a heatmap

    Arguments:
        model (RNN): a built RNN
        ax: axis. Axis to plot onto
        scalebar: bool. If true a scale bar is shown to indicate values
    """
    if ax is None:
        f, ax = plt.subplots(figsize=(10, 10))
    else:
        f = ax.figure

    W = npify(model.get_recurrent_weights(), flatten=False)
    img = ax.imshow(W, cmap="bwr", vmin=-W.max(), vmax=W.max())

    if scalebar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        f.colorbar(img, cax=cax, orientation="vertical")

    ax.set(xticks=[], yticks=[], xlabel="units", ylabel="units")
    ax.axis("equal")
    clean_axes(f)
    return f, ax


def plot_model_weights(model):
    """
    Plots input, recurrent and output weights for a
    given RNN.


    Arguments:
        model (RNN): a built RNN
    """

    f, axes = create_triplot(figsize=(10, 10))

    # plot recurrent weights
    plot_recurrent_weights(model, ax=axes.main, scalebar=False)

    # plot input/output weights
    _in = npify(model.w_in.weight).T
    _out = npify(model.w_out.weight).T
    axes.top.imshow(_in, cmap="bwr", vmin=-_in.max(), vmax=_in.max())
    axes.right.imshow(_out, cmap="bwr", vmin=-_out.max(), vmax=_out.max())

    # clean axes
    axes.main.set(title="Recurrent weights")
    axes.top.set(title="Input weights")
    axes.right.set(title="Output weights")

    axes.main.axis("off")
    axes.top.axis("off")
    axes.right.axis("off")

    return f, axes


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
