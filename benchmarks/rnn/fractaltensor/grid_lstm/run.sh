#!/bin/bash

# overall for figure 7.
./build/grid_rnn 32 256 10 1
./build/grid_rnn 32 512 10 0
./build/grid_rnn 32 1024 10 0

depths='1 2 4 8 16 32'
hiddens='256 1024' # for middle size and large size
# scale with depth in Figure 9
for h in $hiddens; do
    for d in $depths; do
        ./build/grid_rnn $d $h 10 0
    done
done

hiddens='256 1024' # for middle size and large size
# scale with length in Figure 9
lengths='5, 7, 10'
for h in $hiddens; do
    for l in $lengths; do
        ./build/grid_rnn 32 $h $l 0
    done
done
