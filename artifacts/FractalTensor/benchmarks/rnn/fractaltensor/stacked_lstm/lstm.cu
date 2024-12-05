#include "../../cuDNN/RNN_example.h"
#include "../../cuDNN/utils.h"
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

void TransformedDepthFirst(int seq_length, int batch_size, int hidden_size,
                           int depth, std::vector<std::vector<float>>& times,
                           bool fuse_bmm = true) {
    cudaStream_t stream;
    CudaCheck(cudaStreamCreate(&stream));

    auto allocator = std::make_shared<CudaMemoryPool>();
    allocator->add_track_stream(stream);

    CUDAPlace place = CUDAPlace(0);
    GPUContext context(place);

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

    /* Layout of xss is batch major: [batch_size, seq_length, hidden_size]*/
    Region1DepthFirst(context, hsss, csss, xss, ws, us, allocator, times[0]);
    Region2DepthFirst(context, hsss, csss, xss, ws, us, allocator, times[1]);
    Region3DepthFirst(context, hsss, csss, xss, ws, us, allocator, times[2],
                      fuse_bmm);

    times[3][0] = times[0][0] + times[1][0] + times[2][0];
    times[3][1] = times[0][1] + times[1][1] + times[2][1];
    times[3][2] = times[0][2] + times[1][2] + times[2][2];
    times[3][3] = times[0][3] + times[1][3] + times[2][3];

    // ops::PrintOp<GPUContext, CUDAPlace, float> printer;
    // LOG(INFO) << "hsss: " << printer(hsss) << std::endl;
    // LOG(INFO) << "csss: " << printer(csss) << std::endl;
}

void TestDepthFirst(int batch_size, int seq_length, int hidden_size, int depth,
                    std::vector<std::vector<float>>& times, int iter_count) {
    std::vector<std::vector<float>> warmup_times(5, std::vector<float>(4, 0.));
    TransformedDepthFirst(seq_length, batch_size, hidden_size, depth,
                          warmup_times);
    for (int i = 0; i < iter_count; ++i) {
        TransformedDepthFirst(seq_length, batch_size, hidden_size, depth,
                              times);
    }
}
}  // namespace core
}  // namespace kaleido

int main(int argc, char* argv[]) {
    InitGLOG("hello_world");

    int batch_size = 128;
    int seq_length = 50;

    // std::vector<int> seq_lengths = {5, 25, 50, 100, 150, 200, 300, 350, 400};
    // std::vector<int> hidden_sizes = {16, 32, 64, 128, 256, 512};
    std::vector<int> hidden_sizes = {256};
    // std::vector<int> depths = {1};

    std::vector<int> depths(32, 1);
    for (int i = 0; i < 32; ++i) depths[i] = i + 1;

    // int batch_size = 2;
    // int seq_length = 7;
    // int hidden_size = 2;

    printHeader();
    for (auto hidden_size : hidden_sizes) {
        for (auto depth : depths) {
            genSeqs(batch_size, seq_length, false);

            std::stringstream ss;
            ss << "[" << batch_size << ", " << hidden_size << ", " << seq_length
               << ", " << depth << "]\t";

            const int iter_count = 10;
            std::vector<std::vector<float>> times(4, std::vector<float>(4, 0.));
            kaleido::core::TestDepthFirst(batch_size, seq_length, hidden_size,
                                          depth, times, iter_count);

            std::vector<std::string> names(4);
            names[0] = ("(0, >=0, \\*)");
            names[1] = ("(>0, 0, \\*)");
            names[2] = ("(>0, >0, \\*)");
            names[3] = ("Total");
            PrintRecord(ss.str(), names, times, iter_count, false);

            int input_size = hidden_size;
            float cudnn_time = TestCuDNNLSTM(batch_size, hidden_size,
                                             seq_length, depth, input_size);

            std::cout << "CuDNN\t" << ss.str() << "\t\t\t\t" << cudnn_time
                      << std::endl;
        }
    }
}
