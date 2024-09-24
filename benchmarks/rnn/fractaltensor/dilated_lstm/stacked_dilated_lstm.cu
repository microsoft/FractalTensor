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

void TransformedDepthFirst(int batch_size, int seq_length, int hidden_size,
                           int depth, std::vector<std::vector<float>>& times) {
    /* Layout of xss is batch major: [batch_size, seq_length, hidden_size]*/
    cudaStream_t stream;
    CudaCheck(cudaStreamCreate(&stream));

    auto allocator = std::make_shared<CudaMemoryPool>();
    allocator->add_track_stream(stream);

    CUDAPlace place = CUDAPlace(0);
    GPUContext context(place);

    if (depth > 2) {
        int const kMaxLink = 2 << (depth - 2);
        seq_length = (((seq_length - 1) / kMaxLink) + 1) * kMaxLink;
    }

    Tensor xss({seq_length, batch_size, hidden_size}, allocator);
    RandomTensor(xss, 0, 0.01);

    int const kMatCount = 4;
    Tensor ws({depth, hidden_size, kMatCount * hidden_size}, allocator);
    Tensor us({depth, hidden_size, kMatCount * hidden_size}, allocator);
    Tensor hsss({depth, seq_length, batch_size, hidden_size}, allocator);
    Tensor csss({depth, seq_length, batch_size, hidden_size}, allocator);

    RandomTensor(ws, 0, 0.01);
    RandomTensor(us, 0, 0.01);
    FillZeros(hsss);
    FillZeros(csss);

    Region1DepthFirst(context, hsss, csss, xss, ws, us, allocator, times[0]);
    Region2DepthFirstReuse(context, hsss, csss, xss, ws, us, allocator,
                           times[1]);

    times[3][0] = times[0][0] + times[1][0];
    times[3][1] = times[0][1] + times[1][1];
    times[3][2] = times[0][2] + times[1][2];
    times[3][3] = times[0][3] + times[1][3];

    // ops::PrintOp<GPUContext, CUDAPlace, float> printer;
    // std::cout << "hsss: " << printer(hsss) << std::endl;
    // std::cout << "csss: " << printer(csss) << std::endl;
}

void TestDepthFirst(int batch_size, int seq_length, int hidden_size, int depth,
                    std::vector<std::vector<float>>& times, int iter_count) {
    std::vector<std::vector<float>> warmup_times(5, std::vector<float>(4, 0.));
    TransformedDepthFirst(batch_size, seq_length, hidden_size, depth,
                          warmup_times);
    for (int i = 0; i < iter_count; ++i) {
        TransformedDepthFirst(batch_size, seq_length, hidden_size, depth,
                              times);
    }
}
}  // namespace core
}  // namespace kaleido

int main(int argc, char* argv[]) {
    InitGLOG("grid lstm");

    int batch_size = 32;
    int seq_length = 100;
    int depth = 5;
    int hidden_size = 256;

    std::stringstream ss;
    ss << "[" << batch_size << ", " << hidden_size << ", " << seq_length << ", "
       << depth << "]|";

    const int iter_count = 10;
    std::vector<std::vector<float>> times(4, std::vector<float>(4, 0.));
    kaleido::core::TestDepthFirst(batch_size, seq_length, hidden_size, depth,
                                  times, iter_count);

    std::vector<std::string> names(4);
    names[0] = ("(0, >=0, \\*)");
    names[1] = ("(>0, 0, \\*)");
    names[2] = ("(>0, >0, \\*)");
    names[3] = ("Total");
    PrintRecord(ss.str(), names, times, iter_count);

    return 0;
}
