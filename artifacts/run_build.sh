#!/usr/bin/bash
set -e

root_dir=$(pwd)
echo $root_dir

if [ -z $CUDNN_HOME ]; then
    echo "CUDNN_HOME is not set."
    echo "Please set CUDNN_HOME to the location of cuDNN root directory."
    echo "otherwise, the build will fail."
    exit 1
fi

# build the FractalTensor main project
cd FractalTensor
make build CUDNN_HOME=$CUDNN_HOME

benchmark_dir="FractalTensor/benchmarks"

# build the stacked LSTM benchmark
cd $root_dir
cd "$benchmark_dir/rnn/fractaltensor/cute_stacked_lstm/"
make 2>&1 | tee build.log

# build the dilated LSTM benchmark
cd $root_dir
cd "$benchmark_dir/rnn/fractaltensor/cute_dilated_lstm/"
make 2>&1 | tee -a build.log

# build the grid LSTM benchmark
cd $root_dir
cd "$benchmark_dir/rnn/fractaltensor/grid_lstm/"
make 2>&1 | tee -a build.log

# build the back-to-back gemm benchmark
cd $root_dir
cd "$benchmark_dir/fused_two_hgemms/fractaltensor"
make 2>&1 | tee -a build.log

# build the mutli-head attention benchmark
cd $root_dir
cd "$benchmark_dir/multi-head_attention/fractaltensor/"
make 2>&1 | tee -a build.log

# build the bigbird attention benchmark
cd $root_dir
cd "$benchmark_dir/blocked_sparse_attention/fractaltensor"
make 2>&1 | tee -a build.log
