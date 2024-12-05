#!/bin/bash

root_dir=$(pwd)

log_dir="$root_dir/logs"

if [ ! -d "$log_dir" ]; then
    mkdir $log_dir
fi

rnn_bench_dir="FractalTensor/benchmarks/rnn/fractaltensor"

# 1. run the stacked LSTM benchmark
stack_rnn_path="$rnn_bench_dir/cute_stacked_lstm/build/stacked_lstm"
if [ -f "$stack_rnn_path" ]; then
    echo "Running stacked LSTM benchmark"
    $stack_rnn_path $log_dir/ft_stacked_lstm.tsv
else
    echo "Stacked LSTM is not found. Check whether the build is successful."
fi

# 2. run the dilated LSTM benchmark
dilated_rnn_path="$rnn_bench_dir/cute_dilated_lstm/build/dilated_lstm"
if [ -f "$dilated_rnn_path" ]; then
    echo "Running dilated LSTM benchmark"
    $dilated_rnn_path $log_dir/ft_dilated_lstm.tsv
else
    echo "Dilated LSTM is not found. Check whether the build is successful."
fi

# 3. run the grid LSTM benchmark
grid_rnn_dir="$rnn_bench_dir/grid_lstm/"
cd $grid_rnn_dir

if [ -f "build/grid_rnn" ]; then
    echo "Running grid LSTM benchmark"
    sh run.sh 2>&1 | tee $log_dir/ft_grid_lstm.tsv
else
    echo "Grid LSTM is not found. Check whether the build is successful."
fi
cd $root_dir

benchmark_dir="FractalTensor/benchmarks"

# 4. run the mha benchmark
mha_dir="$benchmark_dir/multi-head_attention/fractaltensor/build"
mha_exe="$mha_dir/main"

if [ -f "$mha_exe" ]; then
    echo "Running multi-head attention benchmark"
    $mha_exe 2>&1 | tee $log_dir/ft_attention.tsv
else
    echo "Multi-head attention is not found. Check whether the build is successful."
fi

# 5. run the back-2-back gemm benchmark
b2b_gemm_dir="$benchmark_dir/fused_two_hgemms/fractaltensor/build/"
if [ -f "$b2b_gemm_dir/hgemm_b2b" ]; then
    echo "Running back-2-back GEMM benchmark"
    $b2b_gemm_dir/hgemm_b2b $log_dir/ft_b2b_gemm.tsv
else
    echo "Back-2-back GEMM is not found. Check whether the build is successful."
fi

# 6. run the bigbird benchmark
bigbird_dir="$benchmark_dir/blocked_sparse_attention/fractaltensor/build"
if [ -f "$bigbird_dir/bigbird" ]; then
    echo "Running BigBird benchmark"
    $bigbird_dir/bigbird 2>&1 | tee $log_dir/ft_bigbird.tsv
else
    echo "BigBird is not found. Check whether the build is successful."
fi
