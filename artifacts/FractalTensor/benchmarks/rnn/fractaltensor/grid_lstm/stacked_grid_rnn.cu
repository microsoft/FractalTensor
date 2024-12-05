#include "kaleido/core/cuda_allocator.h"
#include "kaleido/core/operators/print_op.h"
#include "regions/regions.h"

#include <algorithm>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

namespace kaleido {
namespace core {

void TransformedDepthFirst(int batch_size, int src_length, int trg_length,
                           int hidden_size, int depth,
                           std::vector<std::vector<float>>& times) {
    /* Layout of xss is batch major: [batch_size, seq_length, hidden_size]*/

    cudaStream_t stream;
    CudaCheck(cudaStreamCreate(&stream));

    auto allocator = std::make_shared<CudaMemoryPool>();
    allocator->add_track_stream(stream);

    CUDAPlace place = CUDAPlace(0);
    GPUContext context(place);

    Tensor xss({src_length, batch_size, hidden_size}, allocator);
    Tensor yss({trg_length, batch_size, hidden_size}, allocator);
    RandomTensor(xss, 0, 0.01);
    RandomTensor(yss, 0, 0.01);

    int const kGridDim = 2;
    Tensor ws({depth, hidden_size, hidden_size}, allocator);
    Tensor us({depth, kGridDim * hidden_size, hidden_size}, allocator);
    Tensor bs({depth, batch_size, hidden_size}, allocator);
    Tensor hsss(
        {depth, src_length, trg_length, kGridDim, batch_size, hidden_size},
        allocator);

    RandomTensor(ws, 0, 0.01);
    RandomTensor(us, 0, 0.01);
    RandomTensor(bs, 0, 0.01);
    FillZeros(hsss);

    Region1DepthFirst(context, hsss, xss, yss, ws, us, bs, allocator, times[0]);
    Region2DepthFirst(context, hsss, xss, yss, ws, us, bs, allocator, times[1]);
    Region3DepthFirst(context, hsss, xss, yss, ws, us, bs, allocator, times[2]);
    Region4DepthFirst(context, hsss, xss, yss, ws, us, bs, allocator, times[3]);
    Region5DepthFirst(context, hsss, xss, yss, ws, us, bs, allocator, times[4]);
    Region6DepthFirst(context, hsss, xss, yss, ws, us, bs, allocator, times[5]);
    Region7DepthFirst(context, hsss, xss, yss, ws, us, bs, allocator, times[6]);
    Region8DepthFirst(context, hsss, xss, yss, ws, us, bs, allocator, times[7]);

    times[times.size() - 1][0] = 0.;
    times[times.size() - 1][1] = 0.;
    times[times.size() - 1][2] = 0.;
    times[times.size() - 1][3] = 0.;
    for (int i = 0; i < times.size() - 1; ++i) {
        times[times.size() - 1][0] += times[i][0];
        times[times.size() - 1][1] += times[i][1];
        times[times.size() - 1][2] += times[i][2];
        times[times.size() - 1][3] += times[i][3];
    }

    // ops::PrintOp<GPUContext, CUDAPlace, float> printer;
    // std::cout << "hsss: " << printer(hsss) << std::endl;
    // std::cout << "csss: " << printer(csss) << std::endl;
}

}  // namespace core
}  // namespace kaleido

void run_test(int batch_size, int depth, int src_length, int trg_length,
              int hidden_size, bool print_header = true) {
    InitGLOG("grid_lstm");
    const int regions = 8 + 1;  // 8 regions + total

    // warm up run
    std::vector<std::vector<float>> warmup_times(regions,
                                                 std::vector<float>(4, 0.));
    kaleido::core::TransformedDepthFirst(batch_size, src_length, trg_length,
                                         hidden_size, depth, warmup_times);

    std::stringstream ss;
    ss << "[" << batch_size << ", " << hidden_size << ", " << src_length << ", "
       << trg_length << ", " << depth << "]\t";

    const int iter_count = 5;
    std::vector<std::vector<float>> times(regions, std::vector<float>(4, 0.));

    for (int i = 0; i < iter_count; ++i) {
        kaleido::core::TransformedDepthFirst(batch_size, src_length, trg_length,
                                             hidden_size, depth, times);
    }

    std::vector<std::string> names(9);
    names[0] = ("(0, >=0, \\*)");
    names[1] = ("(>0, 0, \\*)");
    names[2] = ("(>0, >0, \\*)");
    names[3] = ("Total");
    PrintRecord(ss.str(), names, times, iter_count, print_header);
}

int main(int argc, char* argv[]) {
    assert(argc == 5);

    int depth = atoi(argv[1]);
    int hidden_size = atoi(argv[2]);
    int length = atoi(argv[3]);
    bool print_header = bool(atoi(argv[4]));

    int batch = 32;

    run_test(batch, depth, length, length, hidden_size, print_header);

    return 0;
}
