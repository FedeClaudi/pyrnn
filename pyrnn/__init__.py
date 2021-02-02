import pyinspect as pi

pi.install_traceback(keep_frames=1)


from pyrnn.rnn import RNN, TorchRNN, CTRNN
from pyrnn import analysis


import sys

is_win = sys.platform == "win32"

from loguru import logger
from rich.logging import RichHandler

logger.configure(
    handlers=[{"sink": RichHandler(markup=True), "format": "{message}"}]
)
