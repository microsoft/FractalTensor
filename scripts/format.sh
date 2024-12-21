#!/bin/bash

#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# Format Python files using yapf
echo "Running yapf..."
find . -type f -name "*.py" \
    ! -path "./build/*" \
    ! -path "./.git/*" \
    ! -path "*.egg-info/*" \
    -print0 | xargs -0 yapf --in-place

# Format Python imports using isort
echo "Running isort..."
isort .



find kaleido/ -name "*.cc" -o -name "*.cpp" -o -name "*.h" -o -name "*.cu" | xargs clang-format -i
find benchmarks/ -name "*.cpp" -o -name "*.h" -o -name "*.cu" | xargs clang-format -i
