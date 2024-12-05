#!/bin/bash
ncu_dir="/home/sosp/env/spack/opt/spack/linux-ubuntu22.04-zen2/gcc-11.4.0/cuda-12.4.0-ypujjdfaen2zwiplopzke4ud33wddscv/bin"

root_dir=$(pwd)
log_dir="$root_dir/logs"
benchmark_dir="FractalTensor/benchmarks"

# 1. ncu test the mha benchmark
echo "NCU profiling mha benchmark"
mha_dir="$benchmark_dir/multi-head_attention/fractaltensor/build"
mha_exe="$mha_dir/main"
$ncu_dir/ncu --section "MemoryWorkloadAnalysis" \
    --metrics "dram__bytes.sum,lts__t_bytes.sum,l1tex__t_bytes.sum" \
    --csv $mha_exe > $log_dir/ft_attention_ncu.csv

# 2. ncu test the bigbird benchmark
bigbird_dir="$benchmark_dir/blocked_sparse_attention/fractaltensor/build"
echo "NCU profiling BigBird benchmark"
$ncu_dir/ncu --section "MemoryWorkloadAnalysis" \
    --metrics "dram__bytes.sum,lts__t_bytes.sum,l1tex__t_bytes.sum" \
    --csv $bigbird_dir/bigbird > $log_dir/ft_bigbird_ncu.csv
