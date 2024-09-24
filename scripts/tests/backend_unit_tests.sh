#!/bin/bash
set -e

# Device Unit Tests
for ut in $(find build/kaleido/core/device/tests/test_*); do
    echo "Running $ut"
    ./$ut
done

# Operators Unit Tests
for ut in $(find build/kaleido/core/operators/tests/test_*); do
    if [[ $ut != *"test_gather_scatter"* && $ut != *"test_gemm_batched"* ]]; then
        echo "Running $ut"
        ./$ut
    else
        echo "Skipping $ut"
    fi
done

for ut in $(find build/kaleido/core/tests/test_*); do
    echo "Running $ut"
    ./$ut
done
