// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../../../build/third_party/tvm/src/extern_tvm/src/runtime/cuda/cuda_device_api.cc"
#include "../../../build/third_party/tvm/src/extern_tvm/src/runtime/cuda/cuda_module.cc"

#include <dlpack/dlpack.h>
#include <tvm/driver/driver_api.h>
#include <tvm/runtime/contrib/papi.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/profiling.h>
#include <tvm/runtime/registry.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

void initArray(DLTensor* input, const size_t size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1., 1.);

    for (size_t i = 0; i < size; ++i)
        static_cast<float*>(input->data)[i] = dist(gen);
}

constexpr int64_t seq_len = 100;
constexpr int64_t batch_size = 128;
constexpr int64_t input_size = 512;
constexpr int64_t hidden_size = 512;

void lstm_cell() {
    DLDevice dev{kDLCUDA, 0};
    // DLDevice dev{kDLCPU, 0};
    std::string lib_path = "../lstm_cell.so";

    bool enabled = tvm::runtime::RuntimeEnabled("cuda");
    const tvm::runtime::PackedFunc* graph_executor_create =
        tvm::runtime::Registry::Get("tvm.graph_executor.create");

    auto lib = tvm::runtime::Module::LoadFromFile(lib_path);

    tvm::runtime::Module gmod = lib.GetFunction("default")(dev);
    tvm::runtime::PackedFunc set_input = gmod.GetFunction("set_input");
    tvm::runtime::PackedFunc get_output = gmod.GetFunction("get_output");
    tvm::runtime::PackedFunc run = gmod.GetFunction("run");

    DLTensor* B;
    DLTensor* C;
    DLTensor* out;
    int ndim = 2;
    int dtype_code = kDLFloat;
    int dtype_bits = 32;
    int dtype_lanes = 1;
    int device_type = kDLCPU;
    int device_id = 0;

    int64_t input_shape[2] = {batch_size, input_size};
    DLTensor* x;
    TVMArrayAlloc(input_shape, 2 /*ndim*/, dtype_code, dtype_bits, dtype_lanes,
                  device_type, device_id, &x);
    initArray(x, batch_size * input_size);

    int64_t state_shape[2] = {batch_size, hidden_size};
    DLTensor* h_prev;
    TVMArrayAlloc(state_shape, 2 /*hdim*/, dtype_code, dtype_bits, dtype_lanes,
                  device_type, device_id, &h_prev);
    initArray(h_prev, batch_size * hidden_size);

    DLTensor* c_prev;
    TVMArrayAlloc(state_shape, 2 /*hdim*/, dtype_code, dtype_bits, dtype_lanes,
                  device_type, device_id, &c_prev);
    initArray(c_prev, batch_size * hidden_size);

    int64_t output_shape[2] = {batch_size, hidden_size};
    DLTensor* y;
    TVMArrayAlloc(output_shape, 2 /*hdim*/, dtype_code, dtype_bits, dtype_lanes,
                  device_type, device_id, &y);
    initArray(y, batch_size * hidden_size);

    set_input("input", x);
    set_input("h_prev", h_prev);
    set_input("c_prev", c_prev);

    run();
    get_output(0, y);

    TVMArrayFree(x);
    TVMArrayFree(h_prev);
    TVMArrayFree(c_prev);
    TVMArrayFree(y);
}

int main() {
    lstm_cell();
    return 0;
}
