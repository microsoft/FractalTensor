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
python3 $stack_rnn_path/stacked_lstm_PyTorch.py \
    --output_file $log_dir/pt_stacked_lstm.tsv \
    --default_test True

# 2. run the dilated LSTM benchmark
dilated_rnn_path="$rnn_bench_dir/stacked_dilated_rnn"

echo "Running dilated LSTM benchmark"
python3 $dilated_rnn_path/stacked_drnn_pytorch.py \
    --output_file $log_dir/pt_dilated_lstm.tsv \
    --default_test True

# 3. run the grid LSTM benchmark
grid_rnn_dir="$rnn_bench_dir/grid_lstm"

echo "Running grid LSTM benchmark"
cd $grid_rnn_dir
./run_grid_lstm_pt.sh 2>&1 | tee $log_dir/pt_grid_lstm.log
cd $root_dir
python3 scripts/post_process_grid_rnn.py

benchmark_dir="FractalTensor/benchmarks"

# 4. run the mha benchmark
mha_dir="$benchmark_dir/multi-head_attention/baseline"
python3 $mha_dir/test_pt_model.py 2>&1 | tee $log_dir/pt_attention.tsv

# 5. run the bigbird benchmark
bigbird_dir="$benchmark_dir/blocked_sparse_attention/pytorch"
echo "Running BigBird benchmark"
python3 $bigbird_dir/main.py 2>&1 | tee $log_dir/pt_bigbird.tsv
