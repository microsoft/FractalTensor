from .rnn import StackedDRNN
import os
import sys
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

__all__ = [
    "StackedDRNN",
]
