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
    --csv python3 $mha_dir/test_pt_model.py > $log_dir/flash2_attention_ncu.csv
