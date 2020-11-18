from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

from pyrnn._utils import flatten_h
from pyrnn._plot import clean_axes


def get_n_components_with_pca(h, variance_th=0.85, plot=True):
    """
    Uses PCA and looks at the variance explained
    by each principal component and looks at how
    many components are needed to reach a given threshold

    Arguments:
        h: tensor with history of hidden state (from rnn.predict_with_history)
        variance_th: float, default .9. The dimensionality of the dynamics
            is given by the number of components necessary to explain this
            fraction of the variance. Should be in range [0, 1]
        plot: bool, default True. If true a plot is made to show
            the fraction of variance explained
    """
    n_units = h.shape[-1]

    # Fit PCA and get number of components to reach variance
    pca = PCA(n_components=n_units).fit(flatten_h(h))
    explained = np.cumsum(pca.explained_variance_ratio_)
    above = np.where(explained > variance_th)[0][0]
    at_above = explained[above]

    # plot
    if plot:
        f, ax = plt.subplots(figsize=(16, 9))

        ax.plot(explained, "-o", lw=2, color="k", label="variance explained")

        ax.plot(
            [0, above],
            [at_above, at_above],
            lw=3,
            color="salmon",
            zorder=-1,
            label="Threshold",
        )
        ax.plot([above, above], [0, at_above], lw=3, color="salmon", zorder=-1)
        ax.scatter(
            above,
            at_above,
            s=100,
            color="w",
            lw=3,
            edgecolors="salmon",
            alpha=0.8,
            zorder=100,
        )

        ax.legend()
        ax.set(
            xticks=[0, above, n_units],
            xticklabels=[1, above + 1, n_units],
            xlabel="Components",
            ylabel="Fraction of variance explained",
        )
        clean_axes(f)

    return above + 1
