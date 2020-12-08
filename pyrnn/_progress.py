from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    ProgressColumn,
)
from datetime import timedelta
import matplotlib.pyplot as plt
from rich.text import Text
import numpy as np
from myterial import (
    orange,
    amber_light,
    teal_light,
    light_blue_light,
    salmon,
)

from pyrnn._plot import clean_axes
import matplotlib

matplotlib.use("TkAgg")  # necessary for plt.ion on windows


"""
    Classes to create fancy progress bars and live loss plotting
"""


class LiveLossPlot:
    def __init__(self, show):
        self.show = show

    def __enter__(self):
        if self.show:
            f, self.ax = plt.subplots(figsize=(7, 4))
            clean_axes(f)
            plt.ion()

        return self

    def _style(self, loss_history):
        self.ax.set(
            title="Training loss",
            ylabel="Loss",
            xlabel="Training epoch",
            ylim=[0, 0.1 + max(loss_history[1:])],
            xticks=[0, np.argmin(loss_history), len(loss_history) - 1],
            yticks=[
                0,
                round(np.min(loss_history), 4),
                loss_history[-1],
                round(np.max(loss_history[1:]), 4),
            ],
        )

    def update(self, loss_history):
        self.ax.clear()
        self.ax.plot(loss_history, lw=3, color=salmon)
        self.ax.scatter(
            np.argmin(loss_history),
            np.min(loss_history),
            color="w",
            lw=2,
            edgecolors=salmon,
            s=200,
            zorder=100,
        )
        self._style(loss_history)

        plt.draw()
        fig = plt.gcf()
        fig.canvas.draw()
        plt.pause(0.0001)

    def __exit__(self, exc_type, exc_val, exc_tb):
        plt.ioff()


# ---------------------------------- Columns --------------------------------- #


class TimeRemainingColumn(ProgressColumn):
    """Renders estimated time remaining."""

    # Only refresh twice a second to prevent jitter
    max_refresh = 0.5

    def render(self, task):
        """Show time remaining."""
        remaining = task.time_remaining
        if remaining is None:
            return Text("-:--:--", style=teal_light)
        remaining_delta = timedelta(seconds=int(remaining))
        return Text("remaining: " + str(remaining_delta), style=teal_light)


class TimeElapsedColumn(ProgressColumn):
    """Renders estimated time elapsed."""

    # Only refresh twice a second to prevent jitter
    max_refresh = 0.5

    def render(self, task):
        """Show time elapsed."""
        elapsed = task.elapsed
        if elapsed is None:
            return Text("-:--:--", style=light_blue_light)
        elapsed_delta = timedelta(seconds=int(elapsed))
        return Text("elapsed: " + str(elapsed_delta), style=light_blue_light)


class SpeedColumn(TextColumn):
    _renderable_cache = {}

    def __init__(self, *args):
        pass

    def render(self, task):
        if task.speed is None:
            return " "
        else:
            return f"{task.speed:.1f} steps/s"


class LossColumn(TextColumn):
    _renderable_cache = {}

    def __init__(self, *args):
        pass

    def render(self, task):
        try:
            return f"[{amber_light}]loss: [bold {orange}]{task.fields['loss']:.6f}"
        except (AttributeError, TypeError):
            return ""


class LearningRateColumn(TextColumn):
    _renderable_cache = {}

    def __init__(self, *args):
        pass

    def render(self, task):
        try:
            return f"[{amber_light}]lr: [bold {orange}]{task.fields['lr']:.6f}"
        except (AttributeError, TypeError):
            return ""


class FPSpeedColumn(TextColumn):
    _renderable_cache = {}

    def __init__(self, *args):
        pass

    def render(self, task):
        if task.fields["fpspeed"] is not None:
            return f"[{amber_light}]fp speed: [bold {orange}]{task.fields['fpspeed']:.6e}"
        else:
            return ""


# ------------------------------- Progress bars ------------------------------ #
# General purpose progress bar
base_progress = Progress(
    "[progress.description]{task.description}",
    BarColumn(bar_width=None),
    "[progress.percentage]{task.percentage:>3.0f}%",
    "•",
    TimeRemainingColumn(),
    TimeElapsedColumn(),
    transient=True,
)

# Progress bar used for training RNNs
train_progress = Progress(
    TextColumn("[bold magenta]Step {task.completed}/{task.total}"),
    SpeedColumn(),
    "•",
    "[progress.description]{task.description}",
    BarColumn(bar_width=None),
    "[progress.percentage]{task.percentage:>3.0f}%",
    "•",
    TimeRemainingColumn(),
    TimeElapsedColumn(),
    "•",
    LossColumn(),
    LearningRateColumn(),
)

# Progress bar used for finding fixed points
fixed_points_progress = Progress(
    "[progress.description]{task.description}",
    BarColumn(bar_width=None),
    "•",
    "[progress.percentage]{task.percentage:>3.0f}%",
    FPSpeedColumn(),
    "•",
    TimeRemainingColumn(),
    TimeElapsedColumn(),
)
