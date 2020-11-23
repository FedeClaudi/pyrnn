import numpy as np
from rich import print
from myterial import (
    purple,
    indigo,
    indigo_dark,
    teal,
    teal_dark,
    light_blue,
    orange,
)

# Fixed points colors
saddle_c = purple
sink_c = indigo
spiral_sink_c = indigo_dark
source_c = teal
spiral_source_c = teal_dark
center_c = light_blue

fp_colors = dict(
    saddle=saddle_c,
    sink=sink_c,
    spiral_sink=spiral_sink_c,
    source=source_c,
    spiral_source=spiral_source_c,
    center=light_blue,
)


def classify_equilibrium(tr, det):
    """
    Classify an equilibriom (e.g. spiral source, sink)
    using the trace determinant method.
    See:
        - https://images.slideplayer.com/39/11002714/slides/slide_10.jpg
        - https://demonstrations.wolfram.com/EigenvaluesAndTheTraceDeterminantPlaneOfALinearMap/

    Arguments:
        det (float): matrix determinant
        tr (float): matrix trace

    Returns:
        str, type of equilibrium
    """

    parabola = (tr ** 2) / 4

    if det < 0:
        equilibrium = "saddle"
    elif det == 0:
        if tr < 0:
            equilibrium = "line of stable fixed points"
        elif tr == 0:
            equilibrium = "uniform"
        else:
            equilibrium = "line of unstable fixed points"
    else:
        if det < parabola:
            if tr < 0:
                equilibrium = "sink"
            else:
                equilibrium = "source"
        elif det == parabola:
            if tr < 0:
                equilibrium = "degenerate sink"
            else:
                equilibrium = "degenerate source"
        else:
            if tr < 0:
                equilibrium = "spiral sink"
            elif tr == 0:
                equilibrium = "center"
            else:
                equilibrium = "spiral source"
    return equilibrium


def get_trc_det(arr):
    """
    Return the trace and determinant of
    an input array.
    """

    trc = np.trace(arr)
    det = np.linalg.det(arr)
    return trc, det


def get_eigs(arr):
    """
    Return the eigenvalues and eigenvectors of
    an input array, sorted by the magnitue
    of the eigenvalues
    """
    eigv, eigvecs = np.linalg.eig(arr)

    if len(set(eigv)) < len(eigv):
        print(f"[{orange}]Found repeated eigenvalues!")

    # Sort by eigv
    sort_idx = np.argsort(eigv)
    eigv = eigv[sort_idx]
    eigvecs = eigvecs[:, sort_idx]

    return eigv, eigvecs
