#!/bin/bash

root_dir=$(pwd)

log_dir="$root_dir/logs"

if [ ! -d "$log_dir" ]; then
    mkdir $log_dir
fi

rnn_bench_dir="FractalTensor/benchmarks/rnn/baselines"

# 1. run the stacked LSTM benchmark (for whileop / graphmode)
stack_rnn_path="$rnn_bench_dir/stacked_lstm"

echo "Running stacked LSTM benchmark for whileop / graphmode"
python3 $stack_rnn_path/stacked_lstm_tensorflow_graph.py \
    --output_file $log_dir/tf1_stacked_lstm.tsv \
    --default_test True

stack_rnn_path="$rnn_bench_dir/stacked_lstm"

# 2. run the stacked LSTM benchmark for auto-graph
echo "Running stacked LSTM benchmark for autograph"
python3 $stack_rnn_path/stacked_lstm_tensorflow_eager.py \
    --output_file $log_dir/tf2_stacked_lstm.tsv \
    --default_test True
