#!/bin/bash
ncu_dir="/home/sosp/env/spack/opt/spack/linux-ubuntu22.04-zen2/gcc-11.4.0/cuda-12.4.0-ypujjdfaen2zwiplopzke4ud33wddscv/bin"

root_dir=$(pwd)
log_dir="$root_dir/logs"
benchmark_dir="FractalTensor/benchmarks"
mha_dir="$benchmark_dir/multi-head_attention/baseline"

# 1. ncu test the mha benchmark
echo "NCU profiling mha benchmark"
$ncu_dir/ncu --section "MemoryWorkloadAnalysis" \
    --metrics "dram__bytes.sum,lts__t_bytes.sum,l1tex__t_bytes.sum" \
    --csv python3 $mha_dir/test_triton_model.py > $log_dir/triton_attention_ncu.csv


bigbird_dir="$benchmark_dir/blocked_sparse_attention/triton"
# 2. ncu test the bigbird benchmark
echo "NCU profiling BigBird benchmark"
$ncu_dir/ncu --section "MemoryWorkloadAnalysis" \
    --metrics "dram__bytes.sum,lts__t_bytes.sum,l1tex__t_bytes.sum" \
    --csv python3 $bigbird_dir/main.py \
    --default_test True > $log_dir/triton_bigbird_ncu.csv
