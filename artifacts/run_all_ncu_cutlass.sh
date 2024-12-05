#!/bin/bash
ncu_dir="/home/sosp/env/spack/opt/spack/linux-ubuntu22.04-zen2/gcc-11.4.0/cuda-12.4.0-ypujjdfaen2zwiplopzke4ud33wddscv/bin"

root_dir=$(pwd)
log_dir="$root_dir/logs"
exe_path="cutlass/build/examples/41_fused_multi_head_attention/41_fused_multi_head_attention_fixed_seqlen"

nheads=8
batch_size=32
head_size=128
length=1024

if [ ! -f cutlass_attn ]; then
    ln -s $exe_path cutlass_attn
fi

# 1. ncu test the mha benchmark
echo "NCU profiling mha benchmark"
$ncu_dir/ncu --section "MemoryWorkloadAnalysis" \
    --metrics "dram__bytes.sum,lts__t_bytes.sum,l1tex__t_bytes.sum" \
    --csv cutlass_attn --nheads="$nheads" \
        --batch_size=$batch_size \
        --head_size=$head_size \
        --head_size_v=$head_size \
        --seq_length="$length" \
        --seq_length_kv=$length \
        --causal=false  > $log_dir/cutlass_attention_ncu.csv
