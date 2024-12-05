#!/bin/bash

root_dir=$(pwd)

log_dir="$root_dir/logs"

if [ ! -d "$log_dir" ]; then
    mkdir $log_dir
fi

rnn_bench_dir="FractalTensor/benchmarks/rnn/tvm"

# 1. run the stacked LSTM benchmark
depths='1 2 4 8 16 32'
hiddens='256 1024'
for h in $hiddens; do
    for d in $depths; do
        output_filename=$log_dir
        python3 $rnn_bench_dir/lstm_tvm_tuned.py \
            --seq_len 64 \
            --hidden_size $h \
            --depth $d \
            2>&1 | tee $log_dir/tvm_stacked_lstm_h${h}_d${d}_.tsv
    done
done
seqs='32 64 128'
hiddens='256 1024'
for h in $hiddens; do
    for s in $seqs; do
        output_filename=$log_dir
        python3 $rnn_bench_dir/lstm_tvm_tuned.py \
            --seq_len $s \
            --hidden_size $h \
            --depth 32 \
            2>&1 | tee $log_dir/tvm_stacked_lstm_h${h}_d${d}_.tsv
    done
done
benchmark_dir="FractalTensor/benchmarks"
# 5. run the back-2-back gemm benchmark
b2b_gemm_dir="$benchmark_dir/fused_two_hgemms/baseline/tvm"

echo "Running back-2-back GEMM benchmark"
python3 $b2b_gemm_dir/fused_two_hgemms.py \
    --output_file $log_dir/tvm_b2b_gemm.tsv

# 6. run the bigbird benchmark
bigbird_dir="$benchmark_dir/blocked_sparse_attention/tvm"
echo "Running BigBird benchmark"
python3 $bigbird_dir/bigbird_tvm.py 2>&1 | tee $log_dir/tvm_bigbird.tsv
