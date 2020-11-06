from rich.progress import (
    Progress,
    BarColumn,
    TimeRemainingColumn,
    TextColumn,
)
from myterial import orange, amber_light


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
        except AttributeError:
            return "no loss"


class LearningRateColumn(TextColumn):
    _renderable_cache = {}

    def __init__(self, *args):
        pass

    def render(self, task):
        try:
            return f"[{amber_light}]lr: [bold {orange}]{task.fields['lr']:.6f}"
        except AttributeError:
            return "no lr"


class FPSpeedColumn(TextColumn):
    _renderable_cache = {}

    def __init__(self, *args):
        pass

    def render(self, task):
        if task.fields["fpspeed"] is not None:
            return f"[{amber_light}]fp speed: [bold {orange}]{task.fields['fpspeed']:.6e}"
        else:
            return ""


train_progress = Progress(
    TextColumn("[bold magenta]Step {task.completed}/{task.total}"),
    SpeedColumn(),
    "•",
    "[progress.description]{task.description}",
    BarColumn(bar_width=None),
    "•",
    "[progress.percentage]{task.percentage:>3.0f}%",
    TimeRemainingColumn(),
    "•",
    LossColumn(),
    LearningRateColumn(),
)


fixed_points_progress = Progress(
    "[progress.description]{task.description}",
    BarColumn(bar_width=None),
    "•",
    "[progress.percentage]{task.percentage:>3.0f}%",
    "•",
    FPSpeedColumn(),
    "•",
    TimeRemainingColumn(),
)
