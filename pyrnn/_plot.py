import seaborn as sns


def clean_axes(f):
    ax_list = f.axes

    for ax in list(ax_list):
        sns.despine(ax=ax, offset=10, trim=False, left=False, right=True)


def points_from_pc(pc):
    return [[pc[i, :], pc[i + 1, :]] for i in range(len(pc) - 1)]
