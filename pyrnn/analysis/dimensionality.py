import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from pyrnn._utils import flatten_h
from pyrnn._plot import clean_axes


class PCA:
    nans_array = (None,)

    def __init__(self, n_components=2):
        self.n_components = n_components
        self.eigen_values, self.eigen_vectors = None, None

    def _remove_nans(self, X):
        """
        Removes nan values from data.
        Returns a copy of the original array
        """
        self.nans_array = np.isnan(X)

        _X = X.copy()
        _X[self.nans_array] = 0
        return _X

    def _add_nans(self, X):
        """
        Adds nans back where self._remove_nans removed them.
        Returns a modifed version of the original array, not a copu
        """
        if self.nans_array is not None:
            X[self.nans_array] = np.nan
            return X
        else:
            return X

    @property
    def variance_explained(self):
        """
        Get the fraction of variance explained by each eigenvalue
        """
        tot = np.sum(self.eigen_values)
        return np.array([eig / tot for eig in self.eigen_values])

    def fit(self, X):
        """
        Fit a PCA embedding to input data X
        """
        logger.debug(
            f"Fitting PCA with {self.n_components} components on array with shape: {X.shape}"
        )
        X = self._remove_nans(X)

        # normalize the data
        # X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        covariance_matrix = np.cov(X.T)
        self.eigen_values, self.eigen_vectors = np.linalg.eig(
            covariance_matrix
        )

        self.eigen_values = np.real(self.eigen_values)
        self.eigen_vectors = np.real(self.eigen_vectors)

        return self

    def transform(self, X):
        """
        Transform some data based on a previously fit embedding
        """
        X = self._remove_nans(X)
        if self.eigen_values is None:
            raise ValueError(
                "You need to fit the PCA model to your data first"
            )

        projection_matrix = (self.eigen_vectors.T[:][: self.n_components]).T

        return self._add_nans(X).dot(projection_matrix)

    def fit_transform(self, X):
        """
        Fit the embedding on some data and then transform them
        """
        self.fit(X)
        return self.transform(X)


def get_n_components_with_pca(
    arr, is_hidden=False, variance_th=0.85, plot=True
):
    """
    Uses PCA and looks at the variance explained
    by each principal component and looks at how
    many components are needed to reach a given threshold

    Arguments:
        arr: 2D np array or tensor with history of hidden state (from rnn.predict_with_history)
        is_hidden: bool. If arr is a history of hidden state this should be set to true
        variance_th: float, default .9. The dimensionality of the dynamics
            is given by the number of components necessary to explain this
            fraction of the variance. Should be in range [0, 1]
        plot: bool, default True. If true a plot is made to show
            the fraction of variance explained
    """
    n_units = arr.shape[-1]

    if is_hidden:
        arr = flatten_h(arr)

    # Fit PCA and get number of components to reach variance
    pca = PCA(n_components=n_units - 1).fit(arr)
    explained = np.cumsum(pca.variance_explained)  # explained_variance_ratio_
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
    else:
        f = None
    return above + 1, f
