#!/bin/bash
set -e

if [ ! -d cutlass ]; then
    echo "cutlass not found. Please build it first."
    exit 1
fi

exe_path="cutlass/build/examples/41_fused_multi_head_attention/41_fused_multi_head_attention_fixed_seqlen"

if [ ! -f $exe_path ]; then
    echo "cutlass attention not found. Please build it first."
    exit 1
fi

if [ ! -f cutlass_attn ]; then
    ln -s $exe_path cutlass_attn
fi

nheads=8
batch_size=32
head_size=128
lengths='128 256 512 768 1024 1536 2048 4096'

logdir="logs"

if [ ! -d $logdir ]; then
    mkdir $logdir
fi

logfile="$logdir/cutlass_attn_128.log"
if [ -f "$logfile" ]; then
    rm $logfile
fi

for length in $lengths; do
    echo "length = ${length}, nheads = ${nheads}, head_size = ${head_size}, batch = ${batch_size}"

    ./cutlass_attn --nheads="$nheads" \
        --batch_size=$batch_size \
        --head_size=$head_size \
        --head_size_v=$head_size \
        --seq_length="$length" \
        --seq_length_kv=$length \
        --causal=false 2>&1 | tee -a $logfile
done

head_size=256

logfile="$logdir/cutlass_attn_256.log"
if [ -f "$logfile" ]; then
    rm $logfile
fi

for length in $lengths; do
    echo "length = ${length}, nheads = ${nheads}, head_size = ${head_size}, batch = ${batch_size}"

    ./cutlass_attn --nheads="$nheads" \
        --batch_size=$batch_size \
        --head_size=$head_size \
        --head_size_v=$head_size \
        --seq_length="$length" \
        --seq_length_kv=$length \
        --causal=false 2>&1 | tee -a $logfile
done
