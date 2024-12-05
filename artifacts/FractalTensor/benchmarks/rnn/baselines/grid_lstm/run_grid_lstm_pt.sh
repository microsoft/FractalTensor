#!/bin/bash

seq_len=10
batch_size=32

# overall
hiddens='256 512 1024'
for hidden in $hiddens; do
    python3 gridlstm_pt.py --seq_len=$seq_len \
        --batch_size=$batch_size \
        --hidden_size=$hidden \
        --depth=32
done

# scale with depth
depths='1 2 4 8 16 32'
hiddens='256 1024'
for hidden in $hiddens; do
    for depth in $depths; do
        python3 gridlstm_pt.py --seq_len=$seq_len \
            --batch_size=$batch_size \
            --hidden_size=$hidden \
            --depth=$depth
    done
done

# scale with length
lengths='5 7 10'
hiddens='256 1024'
for length in $lengths; do
    for hidden in $hiddens; do
        python3 gridlstm_pt.py --seq_len=$seq_len \
            --batch_size=32 \
            --hidden_size=$hidden \
            --depth=32
    done
done
