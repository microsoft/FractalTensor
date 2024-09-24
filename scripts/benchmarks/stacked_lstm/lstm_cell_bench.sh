#!/bin/bash

set -e

echo "Running LstmCell benchmark......"

# Pytorch CuDNN
echo "Pytorch CuDNN LstmCell"
python3 benchmarks/rnn/baselines/stacked_lstm/lstm_cell_pytorch.py benchmarks/rnn/pytorch_cudnn_lstm_cell_bench.tsv

# Cudnn C
echo "C Cudnn LstmCell"
make -C benchmarks/rnn/cuDNN bench BENCH_NAME=lstm_cell_cudnn OUTPUT_FILE=../c_cudnn_lstm_cell_bench.tsv

# CuTe
echo "CuTe LstmCell"
make -C benchmarks/rnn/fractaltensor/cute_stacked_lstm bench BENCH_NAME=lstm_cell OUTPUT_FILE=../../cute_lstm_cell_bench.tsv
