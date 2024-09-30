#include "utils.h"

#include <assert.h>

#include <array>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

int main(int argc, char* argv[]) {
    assert(argc == 2);
    const char* filename = argv[1];

    std::ofstream fout;
    fout.setf(std::ios::fixed);
    fout.precision(4);

    fout.open(filename, std::ios::out);

    srand(1234);
    constexpr std::array<size_t, 5> hidden_sizes = {64, 128, 256, 512, 1024};
    constexpr std::array<size_t, 2> batch_sizes = {32, 64};
    constexpr size_t seq_length = 16;
    constexpr std::array<size_t, 6> depths = {1, 2, 4, 8, 16, 32};

    fout << "[depth, seq_length, batch_size, hidden_size]\tAvgTime(ms)"
         << std::endl;

    for (auto depth : depths) {
        for (auto hidden_size : hidden_sizes) {
            for (auto batch_size : batch_sizes) {
                genSeqs(batch_size, seq_length, false);
                float cudnn_time = TestCuDNNLSTM(
                    batch_size, hidden_size, seq_length, depth, hidden_size);

                fout << "[" << depth << ", " << seq_length << ", " << batch_size
                     << ", " << hidden_size << "]\t" << cudnn_time << std::endl;
            }
        }
    }
}
