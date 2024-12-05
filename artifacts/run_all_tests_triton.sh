#!/bin/bash

root_dir=$(pwd)

log_dir="$root_dir/logs"

if [ ! -d "$log_dir" ]; then
    mkdir $log_dir
fi

rnn_bench_dir="FractalTensor/benchmarks/rnn/baselines"

# 1. run the stacked LSTM benchmark
stack_rnn_path="$rnn_bench_dir/stacked_lstm"

echo "Running stacked LSTM benchmark"
python3 $stack_rnn_path/stacked_lstm_triton.py \
    --output_file $log_dir/triton_stacked_lstm.tsv \
    --default_test True

# 2. run the dilated LSTM benchmark
dilated_rnn_path="$rnn_bench_dir/stacked_dilated_rnn"

echo "Running dilated LSTM benchmark"
python3 $dilated_rnn_path/stacked_drnn_triton.py \
    --output_file $log_dir/triton_dilated_lstm.tsv \
    --default_test True

# 3. run the grid LSTM benchmark
grid_rnn_dir="$rnn_bench_dir/grid_lstm"

echo "Running grid LSTM benchmark"
python3 $grid_rnn_dir/gridlstm_triton.py \
    --output_file $log_dir/triton_grid_lstm.tsv \
    --default_test True

benchmark_dir="FractalTensor/benchmarks"

# 4. run the mha benchmark
mha_dir="$benchmark_dir/multi-head_attention/baseline"

python3 $mha_dir/test_triton_model.py 2>&1 | tee $log_dir/triton_attention.tsv

# 5. run the back-2-back gemm benchmark
b2b_gemm_dir="$benchmark_dir/fused_two_hgemms/baseline/triton"

echo "Running back-2-back GEMM benchmark"
python3 $b2b_gemm_dir/fused_two_hgemms.py \
    --output_file $log_dir/triton_b2b_gemm.tsv

# 6. run the bigbird benchmark
bigbird_dir="$benchmark_dir/blocked_sparse_attention/triton"
echo "Running BigBird benchmark"
python3 $bigbird_dir/main.py \
    --default_test True 2>&1 | tee $log_dir/triton_bigbird.tsv
