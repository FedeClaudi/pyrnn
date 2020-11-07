import matplotlib.pyplot as plt
from myterial import salmon
import numpy as np
from vedo import Lines, show, Sphere, Tube
from sklearn.decomposition import PCA
from rich import print
import networkx as nx
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ._plot import clean_axes, points_from_pc
from ._utils import prepend_dim, npify, flatten_h

from vedo import settings

settings.useDepthPeeling = (
    True  # necessary for rendering of semitransparent actors
)
settings.useFXAA = True  # necessary for rendering of semitransparent actors


# -------------------------------- matplotlib -------------------------------- #


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


# ------------------------------- vedo renders ------------------------------- #
def render(actors, _show=True):
    """
    Render actors in a vedo windows after
    applying a shader style

    Arguments:
        actors (list): list of vedo Mesh instances
        _show (bool): if true the actors are rendred
    """
    for act in actors:
        act.lighting("off")

    if _show:
        print("[green]Render ready")
        show(*actors)


def get_fp_color(n, col_set=1):
    """
    Get the color of a fixed point given
    the number of unstable modes

    Arguments:
        n (int): number of unstable modes
        col_set (int): which colors set to use

    Returns:
        color (str)
    """
    if n == 0:
        color = "seagreen" if col_set == 1 else "lightseagreen"
    elif n == 1:
        color = "salmon" if col_set == 1 else "lightsalmon"
    elif n == 2:
        color = "skyblue" if col_set == 1 else "deepskyblue"
    else:
        color = "magenta" if col_set == 1 else "purple"

    return color


def plot_state_history_pca_3d(
    hidden_history,
    lw=20,
    alpha=0.1,
    color="k",
    _show=True,
    actors=None,
    mark_start=False,
):
    """
    Fits a PCA to high dim hidden state history
    and plots the result in 3d with vedo.

    Arguments:
        hidden_history (np.ndarray): array with history of hidden states
        lw (int): line weight of hidden state trace
        alpha(float): transparency of hidden state trace
        color (str): color of hidden state trace
        _show (bool): if true the actors are rendered
        actors (list): a list of actors to add to the visualisation
        mark_start (bool): if true a spehere is added to
            mark the start of the hidden trace

    Returns:
        pca (PCA): PCA model fit to hidden history
        actors (list): list of actors
    """
    hh = flatten_h(hidden_history)

    pca = PCA(n_components=3).fit(hh)

    pc = pca.transform(hh)
    points = points_from_pc(pc)

    actors = actors or []
    actors.append(Lines(points).lw(lw).alpha(alpha).c(color))

    if mark_start:
        actors.append(Sphere(points[0][0], r=0.15, c=color))

    render(actors, _show=_show)
    return pca, actors


def plot_fixed_points(
    hidden_history,
    fixed_points,
    scale=0.5,
    _show=True,
    fpoint_radius=0.05,
    sequential=False,
    **kwargs,
):
    """
    Plot fixed points on top of a state history
    as colored spheres (colored by number of unstable
    modes) and tubes showing the direction of
    unstable modes in PCA space

    Arguments:
        hidden_history (np.ndarray): array with history of hidden states
        fixed_points (list): list of FixedPoint instances
        scale (float): scale of the tube showing unstable modes
        _show (bool): if True the scene is rendered
        fpoint_radius (float): radius of sphere showing each fixed point
        sequential (bool): if True the FPS are shown in order of increasing
            number of unstable modes

    Returns:
        pca (PCA): PCA model fit to hidden history
        actors (list): list of actors
    """
    # Plot hidden history
    pca, actors = plot_state_history_pca_3d(
        hidden_history, **kwargs, _show=False
    )

    # Specify a transform function
    t = pca.transform

    # Loop over the number of unstable modes
    _vis_actors = actors.copy()
    for n in (0, 1, 2, 3):
        vis_actors = _vis_actors.copy()

        # loop over fixed ponts
        for fp in fixed_points:
            # Get position
            pos = t(fp.h.reshape(1, -1))[0, :]

            # Get color
            n_unstable = fp.n_unstable_modes
            if n_unstable != n and sequential:
                continue

            color = get_fp_color(n_unstable)

            # plot unstable modes
            for stable, eigval, eigvec in fp.eigenmodes:
                if not stable:
                    delta = eigval * -eigvec * scale
                    p0 = pos - t(np.real(delta).reshape(1, -1))
                    p1 = pos + t(np.real(delta).reshape(1, -1))

                    vis_actors.append(
                        Tube(
                            [p1.ravel(), p0.ravel()],
                            c=color,
                            r=fpoint_radius,
                            alpha=1,
                        )
                    )

            # plot fixed points
            vis_actors.append(Sphere(pos, c=color, r=0.1))

        render(vis_actors, _show=_show)
        actors.extend(vis_actors)
        if not sequential:
            break

    return pca, actors


def plot_fixed_points_connectivity_analysis(
    hidden_history,
    fixed_points,
    fps_connectivity,
    _show=True,
    traj_radius=0.01,
    initial_conditions_radius=0.05,
    sequential=False,
    **kwargs,
):
    """
    On top of a fixed points visualisation,
    show the results of running the fixed points
    connectivity analysis (trajectory of each initial condition).


    Arguments:
        hidden_history (np.ndarray): array with history of hidden states
        fixed_points (list): list of FixedPoint instances
        fps_connectivity (list): list with tuples with outcomes
            of running the fixed point connectivity analysis on eahc initial condition.
        _show (bool): if True the scene is rendered
        traj_radius (float): radius of tube used to show each initial condition trajectory
        initial_conditions_radius (float): radius of sphere showing location of
            each initial condition
        sequential (bool): if True trajectories are shown in sequence based on the number
            of unstable modes of the initial condition's FixedPoint.


    Returns:
        pca (PCA): PCA model fit to hidden history
        actors (list): list of actors
    """
    hidden_history = flatten_h(hidden_history)

    pca, actors = plot_fixed_points(
        hidden_history, fixed_points, _show=False, **kwargs
    )

    def t(arr):
        return pca.transform(prepend_dim(arr)).ravel()

    _vis_actors = actors.copy()
    for n in (0, 1, 2, 3):
        vis_actors = _vis_actors.copy()

        for start_fp, end_fp, trajectory in fps_connectivity:
            n_unstable = start_fp.n_unstable_modes
            if n_unstable != n and sequential:
                continue

            # Color based on the number of unstable modes
            color = get_fp_color(n_unstable, col_set=2)

            trajectory = [t(tp) for tp in trajectory]
            vis_actors.append(
                Tube(trajectory, c=color, r=traj_radius, alpha=1)
            )
            vis_actors.append(
                Sphere(trajectory[0], r=initial_conditions_radius, c=color)
            )

        render(vis_actors, _show=_show)
        actors.extend(vis_actors)

        if not sequential:
            break

    return pca, actors


def plot_fixed_points_connectivity_graph(
    hidden_history, fixed_points, graph, edge_radius=0.1, _show=True, **kwargs
):
    """
    Show connections in a directed graph with fixed points
    connectivity as tubes uniting fixed points in PC space.

       Arguments:
            hidden_history (np.ndarray): array with history of hidden states
            fixed_points (list): list of FixedPoint instances
            graph (nx.DiGraph): results of running FixedConnectivity analysis.
                A directed graph showing connections among fixed points
            edge_radius (float): radius of tube used to show graph edges
            _show (bool): if True the scene is rendered


        Returns:
            pca (PCA): PCA model fit to hidden history
            actors (list): list of actors
    """

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
