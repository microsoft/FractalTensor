#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
find kaleido/ -name "*.cpp" -o -name "*.h" -o -name "*.cu" | xargs clang-format -i
find benchmarks/ -name "*.cpp" -o -name "*.h" -o -name "*.cu" | xargs clang-format -i
