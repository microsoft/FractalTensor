# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from .model import StackedDRNN, StackedDRNNJIT

__all__ = [
    'StackedDRNNJIT',
    'StackedDRNN',
]
