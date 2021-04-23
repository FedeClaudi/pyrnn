import numpy as np
from loguru import logger
from myterial import (
    teal_dark,
    light_blue,
    orange,
)

# Fixed points colors
fp_colors = dict(attractor=orange, center=teal_dark, saddle=light_blue)


def classify_equilibrium(eigenvalues):
    """
    Given the eigenvalues of a system
    linearized around an equilibrium,
    classify the type of equilibriums
    """
    real = np.abs(eigenvalues)

    if np.all(real < 1.0):
        equilibrium = "attractor"
    else:
        if np.any(real == 1):
            logger.warning(
                f"[{orange}]Some eigenvalues have purely immaginary components. "
                "They could be centers if the conditions for the Harman-Grobman "
                "theorem are met."
            )
            equilibrium = "center"
        else:
            n_unstable = len(np.where(real > 1)[0])
            equilibrium = f"{n_unstable}-saddle"
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
    an input array, sorted by the magnitude
    of the eigenvalues
    """
    eigv, eigvecs = np.linalg.eig(arr)

    if len(set(eigv)) < len(eigv):
        logger.warning(f"[{orange}]Found repeated eigenvalues!")

    # Sort by eigv
    sort_idx = np.argsort(eigv)
    eigv = eigv[sort_idx]
    eigvecs = eigvecs[:, sort_idx]

    return eigv, eigvecs
