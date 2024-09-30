#include "RNN_example.h"

template <typename T_ELEM>
float runRNNSample(RNNSampleOptions& options) {
    RNNSample<T_ELEM> sample;
    sample.setup(options);
    sample.run();
    return sample.timeForward;
}

float TestCuDNNLSTM(int mini_batch, int hidden_size, int seq_length,
                    int num_layers, int input_size) {
    RNNSampleOptions options;

    options.dataType = 1;  // CUDNN_DATA_FLOAT
    // options.dataType = 0;
    options.seqLength = seq_length;
    options.numLayers = num_layers;
    options.inputSize = input_size;
    options.hiddenSize = hidden_size;
    options.projSize = hidden_size;
    options.miniBatch = mini_batch;
    options.inputMode = 1;      // CUDNN_LINEAR_INPUT
    options.dirMode = 0;        // CUDNN_UNIDIRECTIONAL
    options.cellMode = 2;       // CUDNN_LSTM
    options.biasMode = 3;       // CUDNN_RNN_DOUBLE_BIAS
    options.algorithm = 0;      // CUDNN_RNN_ALGO_STANDARD
    options.mathPrecision = 1;  // CUDNN_DATA_FLOAT
    // options.mathPrecision = 0;
    options.mathType = 0;  // CUDNN_DEFAULT_MATH
    // options.mathType = 1;  // CUDNN_TENSOR_OP_MATH
    options.dropout = 0.;
    options.printWeights = 0;

    return runRNNSample<float>(options);
    // return runRNNSample<__half>(options);
}

int getRand(int min, int max) { return (rand() % (max - min)) + min + 1; }

void genSeqs(int batch_size, int seq_length, bool random) {
    std::vector<int> temp(batch_size, seq_length);

    std::default_random_engine e;
    e.seed(1234);
    std::normal_distribution<float> distribution(seq_length / 2,
                                                 seq_length / 8);

    for (int i = 1; i < batch_size; ++i) {
        if (random) {
            temp[i] = (int)distribution(e);
        } else {
            temp[i] = seq_length;
        }
    }
    sort(temp.begin(), temp.end());
    reverse(temp.begin(), temp.end());
    seqs = temp;
}
