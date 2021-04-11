import matplotlib.pyplot as plt

from pyrnn import CTRNN as RNN
from pyrnn.plot import plot_recurrent_weights
from pyrnn._utils import constrained_connectivity

"""
    Shows how to initialize an RNN with default weights,
    without autopses, with Dale's ration and with connectivity constraints
"""

# create a figure
f, axes = plt.subplots(ncols=2, nrows=2, figsize=(9, 9))
f.tight_layout()

# ------------------------------ default weights ----------------------------- #
default = RNN()
plot_recurrent_weights(default, ax=axes[0, 0])
axes[0, 0].set(title="Default weights")


# -------------------------------- no autopses ------------------------------- #
autopses = RNN(autopses=False)
plot_recurrent_weights(autopses, ax=axes[0, 1])
axes[0, 1].set(title="No autopses")

# -------------------------------- Dale ratio -------------------------------- #
dale = RNN(dale_ratio=0.8)
plot_recurrent_weights(dale, ax=axes[1, 0])
axes[1, 0].set(title="Dale ratio = 0.8")

# ------------------------- connectivity constraints ------------------------- #
# create  constraints matrix
connectivity = constrained_connectivity((30, 20), 0.4, 0.05)

# create RNN with given connectivity
connectivity_rnn = RNN(connectivity=connectivity)
plot_recurrent_weights(connectivity_rnn, ax=axes[1, 1])
axes[1, 1].set(title="Constrained connectivity")

plt.show()
