# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the
# MIT License.
# --------------------------------------------------------------------------

set(TEST_ROOT ${PROJECT_SOURCE_DIR}/kaleido/core/device/tests)

nv_test(test_tile_copy SRCS ${TEST_ROOT}/test_tile_copy.cu)
