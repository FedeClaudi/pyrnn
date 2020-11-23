import os
import sys

sys.path.append("./")
from three_bit_memory import make_batch


from pyrnn import RNN
from pyrnn.analysis.dimensionality import get_n_components_with_pca
from pyrnn.analysis import FixedPoints, list_fixed_points
from pyrnn.plot import (
    plot_eigenvalues,
    plot_eigenvalues_magnitudes,
    plot_fixed_points_eigenvalues,
    plot_fixed_point_classification,
)
from pyrnn.linalg import get_eigs, get_trc_det
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


# ---------------------------------- Set up ---------------------------------- #

n_units = 64
N = 5000
rnn = RNN.load(
    "./3bit_memory.pt", n_units=n_units, input_size=3, output_size=3
)

X, Y = make_batch(N)
o, h = rnn.predict_with_history(X)

# ---------------------- Dimensionality of the dynamics ---------------------- #
"""
    Let's use PCA to see what's a reasonable estimate of the dimensionality
    of the hidden state's dynamics
"""
get_n_components_with_pca(h, is_hidden=True)

"""
    Or we can look at the eigenvalues of the dynamic's jacobian at
    the fixed points
"""
fps = FixedPoints.load_fixed_points("./3bit_fps.json")
list_fixed_points(fps)
plot_fixed_points_eigenvalues(fps, only_dominant=False)


"""
Plot each fixed point in the trace/determinant plane
"""
f, ax = plt.subplots(figsize=(12, 8))
for fp in fps:
    trace, det = get_trc_det(fp.jacobian)

    print(round(trace, 3), det, fp.n_unstable_modes)
    plot_fixed_point_classification(trace, det, ax=ax)

plt.show()

# ------------------ Dimensionality of the recurrent weights ----------------- #
"""
    Okay so it looks like the dynamics are restricted to a 3D subspace.
    What happens if we looks at the eigenvalue of the recurrent weights
    matrix?
"""

eigv, eigvecs = get_eigs(rnn.get_recurrent_weights())

ax = plot_eigenvalues(eigv)
ax.set(title="Recurrent weights matrix eigenvalues")
ax = plot_eigenvalues_magnitudes(eigv)
ax.set(title="Recurrent weights matrix eigenvalues magnitudes")

plt.show()
