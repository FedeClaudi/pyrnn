import pyinspect as pi

pi.install_traceback(keep_frames=1)


from pyrnn.rnn import RNN, TorchRNN
from pyrnn import analysis


import sys

is_win = sys.platform == "win32"
