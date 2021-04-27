import matplotlib.pyplot as plt
from pyrnn import CTRNN as RNN
from pyrnn.plot import plot_model_weights
from pyrnn.connectivity import MultiRegionConnectivity, Region


"""
    Shows how the define the connectivity of both
    the recurrent units and of the inputs and
    outputs layers of an RNN with multiple subregions
"""

# Create multi region connectivity with 3 areas
# called th, mop and bg with different number of units each and different parameters (e.g. Dale's ratio).

th = Region(
    name="th",
    n_units=24,
    dale_ratio=1,  # only excitatory
    autopses=False,
)


mop = Region(
    name="mop",
    n_units=64,
    dale_ratio=0.8,
    autopses=False,
)

bg = Region(
    name="bg",
    n_units=32,
    dale_ratio=0,  # only inhibitory
    autopses=False,
)

mrc = MultiRegionConnectivity(
    th,
    mop,
    bg,
    dale_ratio=0.7,
    autopses=False,
)

# create feedforward connections
mrc.add_projection("th", "mop", 0.2)
mrc.add_projection("th", "bg", 0.8)

# create feedback connections
mrc.add_projection("mop", "th", 0.1)

mrc.add_projection("bg", "mop", 0.2)


# specify which input goes to which region
# for each input to the network call `add_input`
# and specify which regions should recieve it
mrc.add_input("th", "bg")
mrc.add_input("mop")

# specify which output comes from which region
mrc.add_output("mop", "bg")
mrc.add_output("th", "bg")


# create an RNN and visualize model weights
rnn = RNN(
    n_units=mrc.n_units,
    input_size=mrc.n_inputs,
    output_size=mrc.n_outputs,
    connectivity=mrc.W_rec,
    input_connectivity=mrc.W_in,
    output_connectivity=mrc.W_out,
    w_in_bias=True,
)


plot_model_weights(rnn)
plt.show()
