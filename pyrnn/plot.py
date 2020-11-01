import matplotlib.pyplot as plt
from pyinspect._colors import salmon

from ._plot import clean_axes


def plot_training_loss(loss_history):
    f, ax = plt.subplots(figsize=(12, 7))

    ax.plot(loss_history, lw=2, color=salmon)
    ax.set(xlabel="epochs", ylabel="loss", title="Training loss")
    clean_axes(f)
