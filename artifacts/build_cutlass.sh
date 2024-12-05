#!/bin/bash

if [ ! -d "cutlass" ]; then
    git clone https://github.com/NVIDIA/cutlass.git
    cd cutlass
    git checkout f9ece1b
    cd ..
fi

cd cutlass
mkdir build
cd build
cmake ../
make -j32
cd ..
