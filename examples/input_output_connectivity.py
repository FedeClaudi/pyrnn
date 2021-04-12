import matplotlib.pyplot as plt

from pyrnn import CTRNN as RNN
from pyrnn.plot import plot_model_weights
from pyrnn.connectivity import define_connectivity


"""
    Shows how the define the connectivity of both
    the recurrent units and of the inputs and
    outputs layers
"""

cc, ic, oc = define_connectivity(
    n_units=50,
    n_units_a=15,
    a_to_b_p=0.6,
    b_to_a_p=0.05,
    n_inputs=2,
    n_outputs=1,
)


rnn = RNN(
    input_size=2,
    connectivity=cc,
    input_connectivity=ic,
    output_connectivity=oc,
    w_in_bias=True,
)


plot_model_weights(rnn)


plt.show()
