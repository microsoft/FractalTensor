from .rnn import StackedLSTM
import os
import sys
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

__all__ = [
    "StackedLSTM",
]
