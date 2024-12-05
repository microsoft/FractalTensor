#include <cuda.h>
#include <cudnn.h>
#include <stdio.h>
#include <stdlib.h>

#define COUNTOF(arr) int(sizeof(arr) / sizeof(arr[0]))
#define INIT_MEAN 0.0
#define INIT_VAR 0.5
#define WGROUP_COUNT 4

inline void CheckCudaError(cudaError_t code, const char* expr, const char* file,
                           int line) {
    if (code) {
        fprintf(stderr, "ERROR: CUDA error at %s:%d, code=%d (%s) in '%s'\n\n",
                file, line, (int)code, cudaGetErrorString(code), expr);
        exit(1);
    }
}

inline void CheckCudnnError(cudnnStatus_t code, const char* expr,
                            const char* file, int line) {
    if (code) {
        fprintf(stderr, "CUDNN error at %s:%d, code=%d (%s) in '%s'\n\n", file,
                line, (int)code, cudnnGetErrorString(code), expr);
        exit(1);
    }
}

#define CHECK_CUDA_ERR(...)                                            \
    do {                                                               \
        CheckCudaError(__VA_ARGS__, #__VA_ARGS__, __FILE__, __LINE__); \
    } while (0)

#define CHECK_CUDNN_ERR(...)                                            \
    do {                                                                \
        CheckCudnnError(__VA_ARGS__, #__VA_ARGS__, __FILE__, __LINE__); \
    } while (0)

// Returns uniformly distributed integer values between [lower,..,upper],
// ie, with both lower and upper limits included.
inline int RandRangeInt(int lower, int upper) {
    int lo = (lower < upper ? lower : upper);
    int hi = (lower > upper ? lower : upper);
    return lo + int(drand48() * (hi - lo + 1));
}

// Returns uniformly distributed floating point values between [bias,
// bias+range) assuming range>0, ie, including the lower limit but excluding the
// upper bound.
inline double RandRangeDbl(double bias, double range) {
    return range * drand48() + bias;
}

// Initializes buffer with uniformly distributed values with the given mean and
// variance.
template <typename T_ELEM>
void InitBuffer(T_ELEM* image, size_t imageSize, double mean, double var) {
    double range = sqrt(12.0 * var);
    double bias = mean - 0.5 * range;
    for (size_t index = 0; index < imageSize; index++) {
        image[index] = (T_ELEM)RandRangeDbl(bias, range);
    }
}

static char* BaseFile(char* fname) {
    char* base;
    for (base = fname; *fname != '\0'; fname++) {
        if (*fname == '/' || *fname == '\\') {
            base = fname + 1;
        }
    }
    return base;
}
