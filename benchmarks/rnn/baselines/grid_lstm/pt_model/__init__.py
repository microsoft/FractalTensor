import os
import sys
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from .model import StackedGridModel

__all__ = [
    "StackedGridModel",
]
