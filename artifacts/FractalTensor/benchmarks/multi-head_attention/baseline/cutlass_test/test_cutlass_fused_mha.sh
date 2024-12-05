#!/bin/bash

file="model_config.csv"
exec 3<&0
exec 0<$file

batch_size=16
while read line
do
    IFS=',' read -r length layer_num nheads model_dim <<< "$line"
    head_size=$(expr $model_dim / $nheads)
    echo "length = ${length}, nheads = ${nheads}, head_size = ${head_size}, batch = ${batch_size}"
 
    ./41_fused_multi_head_attention_fixed_seqlen \
        --nheads="$nheads" \
        --batch_size=$batch_size \
        --head_size=$head_size \
        --head_size_v=$head_size \
        --seq_length="$length" \
        --seq_length_kv=$length \
        --causal=false
done

exec 0<&3
