import matplotlib.pyplot as plt
from pyinspect._colors import salmon
import numpy as np
from vedo import Lines, show, Spheres, Sphere, Tube
from sklearn.decomposition import PCA
from rich import print
import networkx as nx
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ._plot import clean_axes, points_from_pc
from ._utils import prepend_dim, npify, flatten_h


# -------------------------------- matplotlib -------------------------------- #


def plot_training_loss(loss_history):
    f, ax = plt.subplots(figsize=(12, 7))

    ax.plot(loss_history, lw=2, color=salmon)
    ax.set(xlabel="epochs", ylabel="loss", title="Training loss")
    clean_axes(f)


def plot_recurrent_weights(model):
    f, ax = plt.subplots(figsize=(10, 10))

    img = ax.imshow(npify(model.recurrent_weights, flatten=False), cmap="bwr")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    f.colorbar(img, cax=cax, orientation="vertical")

    ax.set(xticks=[], yticks=[], xlabel="units", ylabel="units")
    ax.axis("equal")
    clean_axes(f)


def plot_fps_graph(graph):
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


# ------------------------------- vedo renders ------------------------------- #
def render(actors, _show=True):
    for act in actors:
        act.lighting("off")

    if _show:
        print("[green]Render ready")
        show(*actors)


def get_fp_color(n, col_set=1):
    if n == 0:
        color = "seagreen" if col_set == 1 else "lightseagreen"
    elif n == 1:
        color = "salmon" if col_set == 1 else "lightsalmon"
    elif n == 2:
        color = "skyblue" if col_set == 1 else "powderblue"
    else:
        color = "magenta" if col_set == 1 else "purple"

    return color


def plot_state_history_pca_3d(
    hidden_history, lw=20, alpha=0.1, color="k", pts=None, _show=True
):
    """
    Fits a PCA to high dim hidden state history
    and plots the result in 3d with vedo
    """
    hidden_history = flatten_h(hidden_history)

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
    hidden_history,
    fixed_points,
    scale=0.5,
    _show=True,
    fpoint_radius=0.05,
    **kwargs,
):
    hidden_history = flatten_h(hidden_history)

    pca, actors = plot_state_history_pca_3d(
        hidden_history, **kwargs, _show=False
    )

    t = pca.transform

    for fp in fixed_points:
        pos = t(fp.h.reshape(1, -1))[0, :]

        color = get_fp_color(fp.n_unstable_modes)
        # plot unstable directions
        for stable, eigval, eigvec in fp.eigenmodes:
            if not stable:
                delta = eigval * -eigvec * scale
                p0 = pos - t(np.real(delta).reshape(1, -1))
                p1 = pos + t(np.real(delta).reshape(1, -1))

                actors.append(
                    Tube(
                        [p1.ravel(), p0.ravel()],
                        c=color,
                        r=fpoint_radius,
                        alpha=1,
                    )
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
    traj_radius=0.01,
    initial_conditions_radius=0.05,
    **kwargs,
):
    hidden_history = flatten_h(hidden_history)

    pca, actors = plot_fixed_points(
        hidden_history, fixed_points, scale=0.5, _show=False, **kwargs
    )

    def t(arr):
        return pca.transform(prepend_dim(arr)).ravel()

    for start_fp, end_fp, trajectory in fps_connectivity:
        # Color based on the number of unstable modes
        color = get_fp_color(start_fp.n_unstable_modes, col_set=2)

        trajectory = [t(tp) for tp in trajectory]
        actors.append(Tube(trajectory, c=color, r=traj_radius, alpha=1))
        actors.append(
            Sphere(trajectory[0], r=initial_conditions_radius, c=color)
        )

    render(actors, _show=_show)
    return pca, actors


def plot_fixed_points_connectivity_graph(
    hidden_history, fixed_points, graph, edge_radius=0.1, _show=True, **kwargs
):
    pca, actors = plot_fixed_points(
        hidden_history, fixed_points, scale=0.5, _show=False, **kwargs
    )

    def t(arr):
        return pca.transform(prepend_dim(arr)).ravel()

    for fp1, fp2, data in graph.edges(data=True):
        p1 = t(data["fp1"].h)
        p2 = t(data["fp2"].h)
        color = get_fp_color(data["fp1"].n_unstable_modes, col_set=2)

        actors.append(
            Tube([p1, p2], r=edge_radius, c=color, alpha=data["prob"])
        )

    render(actors, _show=_show)
    return pca, actors
