import os
import sys
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from .model import StackedDRNN

__all__ = [
    'StackedDRNN',
]
