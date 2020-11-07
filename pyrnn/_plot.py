import seaborn as sns


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
