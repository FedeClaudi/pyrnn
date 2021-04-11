import seaborn as sns
import math
import matplotlib.pyplot as plt
from matplotlib import gridspec
from collections import namedtuple


def create_triplot(**kwargs):
    """
    Creates a figure with one main plot and two plots on the sides
    """
    fig = plt.figure(**kwargs)
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4])
    ax0 = plt.subplot(gs[1, 0])
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[1, 1])
    plt.tight_layout()

    axes = namedtuple("axes", "main top right")
    return fig, axes(ax0, ax1, ax2)


def center_axes(ax):
    """
    Makes a plot's axes meet at the
    center instead of at the bottom left corner
    """
    ax.spines["bottom"].set_position("zero")
    ax.spines["left"].set_position("zero")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    return ax


def calc_nrows_ncols(N, aspect=(16, 9)):
    """
    Computs the number of rows and columns to fit
    a given number N of subplots in a figure with
    aspect `aspect`.
    from: https://stackoverflow.com/questions/36482328/how-to-use-a-python-produce-as-many-subplots-of-arbitrary-size-as-necessary-acco
    """
    width = aspect[0]
    height = aspect[1]
    area = width * height * 1.0
    factor = (N / area) ** (1 / 2.0)
    cols = math.floor(width * factor)
    rows = math.floor(height * factor)
    rowFirst = width < height
    while rows * cols < N:
        if rowFirst:
            rows += 1
        else:
            cols += 1
        rowFirst = not (rowFirst)
    return rows, cols


def clean_axes(f):
    """
    Makes the axes of a matplotlib figure look better
    """
    ax_list = f.axes

    for ax in list(ax_list):
        sns.despine(ax=ax, offset=10, trim=False, left=False, right=True)


def points_from_pc(pc):
    """
    Given a np.array with the results of applying
    PCA to an array (e.g. with h trajectory), return a list of points
    along the trajectory of PCs
    """
    return [[pc[i, :], pc[i + 1, :]] for i in range(len(pc) - 1)]
