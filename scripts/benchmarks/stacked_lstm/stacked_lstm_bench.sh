#!/bin/bash

set -e

echo "Running Stacked Lstm benchmark......"

# Cudnn C
echo "C Cudnn Stacked Lstm"
make -C benchmarks/rnn/cuDNN bench BENCH_NAME=stacked_lstm_cudnn OUTPUT_FILE=../c_cudnn_stacked_lstm_bench.tsv

# CuTe
echo "CuTe Stacked Lstm"
make -C benchmarks/rnn/fractaltensor/cute_stacked_lstm bench BENCH_NAME=stacked_lstm OUTPUT_FILE=../../cute_stacked_lstm_bench.tsv
