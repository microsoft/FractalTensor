#include "utils.h"

int main(int argc, char* argv[]) {
    srand(1234);
    int batch_size = 64;
    int hidden_size = 256;
    int seq_length = 100;
    int depth = 10;

    int input_size = hidden_size;

    genSeqs(batch_size, seq_length, false);

    for (auto depth : {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22}) {
        float cudnn_time = TestCuDNNLSTM(batch_size, hidden_size, seq_length,
                                         depth, input_size);

        std::stringstream ss;
        ss << "[" << batch_size << ", " << hidden_size << ", " << seq_length
           << ", " << depth << "]|";
        std::cout << "|CuDNN|" << ss.str() << "||||" << cudnn_time << "|"
                  << std::endl;
    }

    std::cout << std::endl;

    for (auto seq_length : {50, 75, 100, 125, 150, 175, 200}) {
        genSeqs(batch_size, seq_length, false);
        float cudnn_time = TestCuDNNLSTM(batch_size, hidden_size, seq_length,
                                         depth, input_size);

        std::stringstream ss;
        ss << "[" << batch_size << ", " << hidden_size << ", " << seq_length
           << ", " << depth << "]|";
        std::cout << "|CuDNN|" << ss.str() << "||||" << cudnn_time << "|"
                  << std::endl;
    }

    std::cout << std::endl;

    genSeqs(batch_size, seq_length, true);

    for (auto depth : {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22}) {
        float cudnn_time = TestCuDNNLSTM(batch_size, hidden_size, seq_length,
                                         depth, input_size);

        std::stringstream ss;
        ss << "[" << batch_size << ", " << hidden_size << ", " << seq_length
           << ", " << depth << "]|";
        std::cout << "|CuDNN|" << ss.str() << "||||" << cudnn_time << "|"
                  << std::endl;
    }

    std::cout << std::endl;

    for (auto seq_length : {50, 75, 100, 125, 150, 175, 200}) {
        genSeqs(batch_size, seq_length, true);
        float cudnn_time = TestCuDNNLSTM(batch_size, hidden_size, seq_length,
                                         depth, input_size);

        std::stringstream ss;
        ss << "[" << batch_size << ", " << hidden_size << ", " << seq_length
           << ", " << depth << "]|";
        std::cout << "|CuDNN|" << ss.str() << "||||" << cudnn_time << "|"
                  << std::endl;
    }

    return 0;
}
