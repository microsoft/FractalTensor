# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import sys
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import random
import unittest

import kaleido

from kaleido import Tensor
from kaleido import TensorStorage

from kaleido import FractalTensor
from kaleido import FractalTensorStorage
