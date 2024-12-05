#include "utils.h"

#include <ctype.h>
#include <stddef.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/sysinfo.h>

struct TestOpts {
    TestOpts() { memset(this, 0, sizeof(*this)); }
    int attn_train;
    int attn_data_type;
    int attn_query_map;
    int attn_num_heads;
    int attn_batch_size;
    int attn_beam_size;
    double attn_sm_scaler;
    float attn_dropout_rate;
    int attn_q_size;
    int attn_k_size;
    int attn_v_size;
    int attn_proj_q_size;
    int attn_proj_k_size;
    int attn_proj_v_size;
    int attn_proj_o_size;
    int attn_seq_len_q;
    int attn_seq_len_k;
    int attn_data_layout;
    int attn_res_link;
    int attn_sweep;
    int attn_rand_geom;
    int attn_rand_seed;
};

static void ParseAttnParameters(int argc, char** argv, TestOpts* opts) {
    struct cmdParams {
        const char* name;
        const char* fmt;
        size_t offs;
        const char* desc;
    } param[] = {
        {"attn_train", "%d", offsetof(TestOpts, attn_train),
         "selects API mode (0-inference, 1-training)"},
        {"attn_data_type", "%d", offsetof(TestOpts, attn_data_type),
         "selects data format (0-FP32, 1-FP64)"},
        {"attn_num_heads", "%d", offsetof(TestOpts, attn_num_heads),
         "number of attenton heads"},
        {"attn_batch_size", "%d", offsetof(TestOpts, attn_batch_size),
         "batch size for Q, R, K, V and O arguments"},
        {"attn_beam_size", "%d", offsetof(TestOpts, attn_beam_size),
         "number of sentence candidates in Q, R inputs"},
        {"attn_sm_scaler", "%lg", offsetof(TestOpts, attn_sm_scaler),
         "softmax smoothing or sharpening coefficient"},
        {"attn_dropout_rate", "%g", offsetof(TestOpts, attn_dropout_rate),
         "dropout rate settings applied during training"},
        {"attn_q_size", "%d", offsetof(TestOpts, attn_q_size),
         "original vector length for 'queries'"},
        {"attn_k_size", "%d", offsetof(TestOpts, attn_k_size),
         "original vector length for 'keys'"},
        {"attn_v_size", "%d", offsetof(TestOpts, attn_v_size),
         "original vector length for 'values'"},
        {"attn_proj_q_size", "%d", offsetof(TestOpts, attn_proj_q_size),
         "length of 'queries' vector after projection"},
        {"attn_proj_k_size", "%d", offsetof(TestOpts, attn_proj_k_size),
         "length of 'keys' vector after projection"},
        {"attn_proj_v_size", "%d", offsetof(TestOpts, attn_proj_v_size),
         "length of 'values' vector after projection"},
        {"attn_proj_o_size", "%d", offsetof(TestOpts, attn_proj_o_size),
         "length of 'output' vector after projection"},
        {"attn_seq_len_q", "%d", offsetof(TestOpts, attn_seq_len_q),
         "largest sequence length for Q, R, O arguments"},
        {"attn_seq_len_k", "%d", offsetof(TestOpts, attn_seq_len_k),
         "largest sequence length for K, V arguments"},
        {"attn_data_layout", "%d", offsetof(TestOpts, attn_data_layout),
         "data layout for Q, K, V, O inputs"},
        {"attn_res_link", "%d", offsetof(TestOpts, attn_res_link),
         "enable/disable residual connections"},
        {"attn_sweep", "%d", offsetof(TestOpts, attn_sweep),
         "sweep all time-steps in one inference API call"},
        {"attn_rand_geom", "%d", offsetof(TestOpts, attn_rand_geom),
         "randomize attention task dimensions"},
        {"attn_rand_seed", "%d", offsetof(TestOpts, attn_rand_seed),
         "seed for the random number generator"},
    };

    if (argc == 1) {
        printf("This is the cuDNN multi-head attention API test.\n\n");
        printf("Usage: ./%s [OPTIONS]\n\nProgram options:\n\n",
               BaseFile(*argv));

        for (int i = 0; i < COUNTOF(param); i++) {
            char buf[64];
            sprintf(buf, "-%s<%s>", param[i].name, param[i].fmt);
            printf("%-20s - %s\n", buf, param[i].desc);
        }
        printf("\n");

        exit(-1);
    }

    while (argc > 1) {
        argc--;
        argv++;

        int i;

        for (i = 0; i < COUNTOF(param); i++) {
            const char* pname = param[i].name;
            size_t plen = strlen(pname);
            if (strncmp(*argv + 1, pname, plen) == 0) {
                int count = sscanf(*argv + plen + 1, param[i].fmt,
                                   (char*)opts + param[i].offs);
                if (count != 1) {
                    fprintf(
                        stderr,
                        "ERROR: missing numerical argument in option '%s'\n\n",
                        *argv);
                    exit(-1);
                }
                break;
            }
        }

        if (i >= COUNTOF(param)) {
            fprintf(stderr, "ERROR: unknown switch '%s'\n\n", *argv);
            exit(-1);
        }
    }
}

struct AttnConfig {
    AttnConfig() {
        memset(this, 0, sizeof(*this));
    }  // sets query_map=ALL_TO_ONE

    cudnnAttnQueryMap_t query_map;  // query_map mode

    int num_heads;       // number of attention heads
    int beam_size;       // number of candidates of the same sentence
    double sm_scaler;    // softmax smoothing or sharpening coefficient
    float dropout_rate;  // dropout probability
    int q_size;          // original vector length of "queries"
    int k_size;          // original vector length of "keys"
    int v_size;          // original vector length of "values"
    int q_proj_size;     // "queries" after projection (0=no projection)
    int k_proj_size;     // "keys" after projection (0=no projection)
    int v_proj_size;     // "values" after projection (0=no projection)
    int o_proj_size;     // "output" after projection (0=no projection)
    int seq_len_q;       // max seq length for Q, R, O buffers
    int seq_len_k;       // max seq length for K, V buffers
    int batch_size;      // batch size for Q, R, K, V, O buffers
    bool res_link;       // enable/disable residual connections
    int sweep;           // sweep all time-steps in inference mode
    int rand_geom;       // randomize problem dimensions
    int rand_seed;       // random number generator seed

    // Attention window boundaries for every time-step.
    int* lo_win_idx;
    int* hi_win_idx;

    // Query and key sequence lengths (for each batch/beam sentence).
    int* q_seq_len;
    int* k_seq_len;

    int data_layout;  // data layout, map to one of 6 possible dataAxes
    cudnnSeqDataAxis_t
        data_axes[CUDNN_SEQDATA_DIM_COUNT];  // data order for T, N, and B dim

    cudnnDataType_t data_type;  // data type for Q,K,V inputs, weights, output
    cudnnDataType_t comp_prec;  // compute precision

    int q_length() {
        return this->q_proj_size > 0 ? this->q_proj_size : this->q_size;
    }

    int k_length() {
        return this->k_proj_size > 0 ? this->k_proj_size : this->k_size;
    }

    int v_length() {
        return this->v_proj_size > 0 ? this->v_proj_size : this->v_size;
    }

    int o_length() {
        return this->o_proj_size > 0 ? this->o_proj_size
                                     : this->v_length() * this->num_heads;
    }

    size_t qo_tokens() {
        return size_t(this->seq_len_q) * this->batch_size * this->beam_size;
    }

    size_t kv_tokens() {
        size_t t = size_t(this->seq_len_k) * this->batch_size;
        if (this->query_map == CUDNN_ATTN_QUERYMAP_ONE_TO_ONE) {
            t *= this->beam_size;
        }
        return t;
    }

    size_t q_all_data() { return this->qo_tokens() * this->q_size; }

    size_t k_all_data() { return this->kv_tokens() * this->k_size; }

    size_t v_all_data() { return this->kv_tokens() * this->v_size; }

    size_t o_all_data() { return this->qo_tokens() * this->o_length(); }

    size_t q_all_weights() {
        size_t q_weights =
            (this->q_proj_size > 0 ? size_t(this->q_size) * this->q_proj_size
                                   : 0);
        return q_weights * this->num_heads;
    }

    size_t k_all_weights() {
        size_t k_weights =
            (this->k_proj_size > 0 ? size_t(this->k_size) * this->k_proj_size
                                   : 0);
        return k_weights * this->num_heads;
    }

    size_t v_all_weights() {
        size_t v_weights =
            (this->v_proj_size > 0 ? size_t(this->v_size) * this->v_proj_size
                                   : 0);
        return v_weights * this->num_heads;
    }

    size_t o_all_weights() {
        size_t o_weights = (this->o_proj_size > 0
                                ? size_t(this->v_length()) * this->o_proj_size
                                : 0);
        return o_weights * this->num_heads;
        return o_weights * this->num_heads;
    }

    size_t q_seq_len_count() { return this->batch_size * this->beam_size; }

    size_t k_seq_len_count() {
        return this->batch_size *
               (this->query_map == CUDNN_ATTN_QUERYMAP_ONE_TO_ONE
                    ? this->beam_size
                    : 1);
    }
};

template <typename T_ELEM, typename T_MATH>
class MultiheadAttentionTest {
   public:
    cudnnHandle_t handle;

    AttnConfig main_cfg;

    cudnnAttnDescriptor_t attn_desc;
    cudnnDropoutDescriptor_t drop_desc;
    cudnnSeqDataDescriptor_t q_desc;
    cudnnSeqDataDescriptor_t k_desc;
    cudnnSeqDataDescriptor_t v_desc;
    cudnnSeqDataDescriptor_t o_desc;

    // Attention in/out buffers on the GPU side.
    T_ELEM* dev_q;
    T_ELEM* dev_k;
    T_ELEM* dev_v;
    T_ELEM* dev_o;
    T_ELEM* dev_w;

    // Buffers with in/out data and weights on the CPU side.
    T_ELEM* host_q;
    T_ELEM* host_k;
    T_ELEM* host_v;
    T_ELEM* host_o;
    T_ELEM* host_w;

    // Work-space and reserve-space GPU buffers required by API.
    T_MATH* dev_wk_space;
    T_MATH* dev_reserve;

    // Capacity of weight/wkspace/reserve buffers (in bytes).
    size_t max_weights;
    size_t max_wk_space;
    size_t max_reserve;

    // Capacity of each "seq" data container (in elements).
    size_t max_elem_q;
    size_t max_elem_k;
    size_t max_elem_v;
    size_t max_elem_o;
    size_t max_elem_a;

    size_t max_elem_q_bar;
    size_t max_elem_k_bar;
    size_t max_elem_v_bar;
    size_t max_elem_h_bar;

    // Dropout descriptor settings.
    size_t dropout_buf_size;
    void* dropout_buf;

    // Sequence length arrays for Q,R,O and K,V.
    int* q_seq_array;
    int* k_seq_array;

    int* dev_q_seq_array;
    int* dev_k_seq_array;

    // Attention window.
    int* lo_win_idx;
    int* hi_win_idx;

    void SetUp(TestOpts& opts);

    void Run();

    void TearDown(void);

    void TestGen(AttnConfig* test_desc, bool debug_info = false);
};

template <typename T_ELEM, typename T_MATH>
void MultiheadAttentionTest<T_ELEM, T_MATH>::SetUp(TestOpts& opts) {
    attn_desc = NULL;
    drop_desc = NULL;
    q_desc = NULL;
    k_desc = NULL;
    v_desc = NULL;
    o_desc = NULL;

    dropout_buf = NULL;
    dropout_buf_size = 0;

    dev_q = NULL;
    dev_k = NULL;
    dev_v = NULL;
    dev_o = NULL;
    dev_w = NULL;

    host_q = NULL;
    host_k = NULL;
    host_v = NULL;
    host_o = NULL;
    host_w = NULL;

    dev_wk_space = NULL;
    dev_reserve = NULL;

    max_weights = 0;
    max_wk_space = 0;
    max_reserve = 0;

    max_elem_q = 0;
    max_elem_k = 0;
    max_elem_v = 0;
    max_elem_o = 0;

    q_seq_array = NULL;
    k_seq_array = NULL;

    lo_win_idx = NULL;
    hi_win_idx = NULL;

    main_cfg.num_heads = opts.attn_num_heads;
    main_cfg.batch_size = opts.attn_batch_size;
    main_cfg.beam_size = opts.attn_beam_size;
    main_cfg.sm_scaler = opts.attn_sm_scaler;
    main_cfg.dropout_rate = opts.attn_dropout_rate;
    main_cfg.q_size = opts.attn_q_size;
    main_cfg.k_size = opts.attn_k_size;
    main_cfg.v_size = opts.attn_v_size;
    main_cfg.q_proj_size = opts.attn_proj_q_size;
    main_cfg.k_proj_size = opts.attn_proj_k_size;
    main_cfg.v_proj_size = opts.attn_proj_v_size;
    main_cfg.o_proj_size = opts.attn_proj_o_size;
    main_cfg.seq_len_q = opts.attn_seq_len_q;
    main_cfg.seq_len_k = opts.attn_seq_len_k;
    main_cfg.res_link = opts.attn_res_link == 0 ? false : true;
    main_cfg.sweep = opts.attn_sweep;
    main_cfg.rand_geom = opts.attn_rand_geom != 0 ? 1 : 0;
    main_cfg.rand_seed = opts.attn_rand_seed;
    main_cfg.data_type = cudnnDataType_t(opts.attn_data_type);
    main_cfg.comp_prec = main_cfg.data_type;

    if (main_cfg.num_heads <= 0 || main_cfg.batch_size <= 0 ||
        main_cfg.beam_size <= 0) {
        fprintf(
            stderr,
            "ERROR: wrong attention NumHeads/BatchSize/BeamSize arguments\n\n");
        exit(-1);
    }

    int q_proj_len = main_cfg.q_length();
    int k_proj_len = main_cfg.k_length();
    int out_len = main_cfg.o_length();

    main_cfg.data_layout = opts.attn_data_layout;

    switch (main_cfg.data_layout) {
        case 0:  // data_axes = [T, N, B]
            main_cfg.data_axes[0] = CUDNN_SEQDATA_TIME_DIM;
            main_cfg.data_axes[1] = CUDNN_SEQDATA_BATCH_DIM;
            main_cfg.data_axes[2] = CUDNN_SEQDATA_BEAM_DIM;
            break;

        case 1:  // data_axes = [T, B, N]
            main_cfg.data_axes[0] = CUDNN_SEQDATA_TIME_DIM;
            main_cfg.data_axes[1] = CUDNN_SEQDATA_BEAM_DIM;
            main_cfg.data_axes[2] = CUDNN_SEQDATA_BATCH_DIM;
            break;

        case 2:  // data_axes = [N, T, B]
            main_cfg.data_axes[0] = CUDNN_SEQDATA_BATCH_DIM;
            main_cfg.data_axes[1] = CUDNN_SEQDATA_TIME_DIM;
            main_cfg.data_axes[2] = CUDNN_SEQDATA_BEAM_DIM;
            break;

        case 3:  // data_axes = [N, B, T]
            main_cfg.data_axes[0] = CUDNN_SEQDATA_BATCH_DIM;
            main_cfg.data_axes[1] = CUDNN_SEQDATA_BEAM_DIM;
            main_cfg.data_axes[2] = CUDNN_SEQDATA_TIME_DIM;
            break;

        case 4:  // data_axes = [B, T, N]
            main_cfg.data_axes[0] = CUDNN_SEQDATA_BEAM_DIM;
            main_cfg.data_axes[1] = CUDNN_SEQDATA_TIME_DIM;
            main_cfg.data_axes[2] = CUDNN_SEQDATA_BATCH_DIM;
            break;

        case 5:  // data_axes = [B, N, T]
            main_cfg.data_axes[0] = CUDNN_SEQDATA_BEAM_DIM;
            main_cfg.data_axes[1] = CUDNN_SEQDATA_BATCH_DIM;
            main_cfg.data_axes[2] = CUDNN_SEQDATA_TIME_DIM;
            break;

        default:
            fprintf(stderr, "ERROR: wrong -attn_data_layout%d option\n\n",
                    opts.attn_data_layout);
            exit(-1);
    }
    main_cfg.data_axes[3] = CUDNN_SEQDATA_VECT_DIM;

    CHECK_CUDNN_ERR(cudnnCreate(&handle));
    CHECK_CUDNN_ERR(cudnnCreateAttnDescriptor(&attn_desc));
    CHECK_CUDNN_ERR(cudnnCreateDropoutDescriptor(&drop_desc));
    CHECK_CUDNN_ERR(cudnnCreateSeqDataDescriptor(&q_desc));
    CHECK_CUDNN_ERR(cudnnCreateSeqDataDescriptor(&k_desc));
    CHECK_CUDNN_ERR(cudnnCreateSeqDataDescriptor(&v_desc));
    CHECK_CUDNN_ERR(cudnnCreateSeqDataDescriptor(&o_desc));

    size_t max_q_tokens =
        size_t(main_cfg.seq_len_q) * main_cfg.batch_size * main_cfg.beam_size;
    size_t max_k_tokens = size_t(main_cfg.seq_len_k) * main_cfg.batch_size;

    // Buffer Q/K/V/O capacity in elements.
    max_elem_q = max_q_tokens * main_cfg.q_size;
    max_elem_k = max_k_tokens * main_cfg.k_size;
    max_elem_v = max_k_tokens * main_cfg.v_size;
    max_elem_o = max_q_tokens * out_len;
    max_elem_a = max_q_tokens * main_cfg.num_heads * main_cfg.seq_len_k;

    max_elem_q_bar = max_q_tokens * main_cfg.num_heads * main_cfg.q_proj_size;
    max_elem_k_bar = max_k_tokens * main_cfg.num_heads * main_cfg.k_proj_size;
    max_elem_v_bar = max_k_tokens * main_cfg.num_heads * main_cfg.v_proj_size;
    max_elem_h_bar = max_q_tokens * main_cfg.num_heads * main_cfg.v_proj_size;

    // Allocate input and output buffers (forward/inference pass).
    CHECK_CUDA_ERR(cudaMalloc((void**)&dev_q, max_elem_q * sizeof(T_ELEM)));
    CHECK_CUDA_ERR(cudaMalloc((void**)&dev_k, max_elem_k * sizeof(T_ELEM)));
    CHECK_CUDA_ERR(cudaMalloc((void**)&dev_v, max_elem_v * sizeof(T_ELEM)));
    CHECK_CUDA_ERR(cudaMalloc((void**)&dev_o, max_elem_o * sizeof(T_ELEM)));

    // Allocate input and output buffers (backward/training pass).
    CHECK_CUDNN_ERR(cudnnDropoutGetStatesSize(handle, &dropout_buf_size));
    CHECK_CUDA_ERR(cudaMalloc((void**)&dropout_buf, dropout_buf_size));

    CHECK_CUDNN_ERR(
        cudnnSetDropoutDescriptor(drop_desc, handle, main_cfg.dropout_rate,
                                  dropout_buf, dropout_buf_size, 0));

    CHECK_CUDNN_ERR(cudnnSetAttnDescriptor(
        attn_desc, main_cfg.query_map, main_cfg.num_heads, main_cfg.sm_scaler,
        main_cfg.data_type, main_cfg.comp_prec, CUDNN_DEFAULT_MATH, drop_desc,
        NULL, main_cfg.q_size, main_cfg.k_size, main_cfg.v_size,
        main_cfg.q_proj_size, main_cfg.k_proj_size, main_cfg.v_proj_size,
        main_cfg.o_proj_size, main_cfg.seq_len_q, main_cfg.seq_len_k,
        main_cfg.batch_size, main_cfg.beam_size));

    CHECK_CUDNN_ERR(cudnnGetMultiHeadAttnBuffers(
        handle, attn_desc, &max_weights, &max_wk_space, NULL));

    if (max_weights > 0) {
        CHECK_CUDA_ERR(cudaMalloc((void**)&dev_w, max_weights));
    }
    if (max_wk_space > 0) {
        CHECK_CUDA_ERR(cudaMalloc((void**)&dev_wk_space, max_wk_space));
    }
    if (max_reserve > 0) {
        CHECK_CUDA_ERR(cudaMalloc((void**)&dev_reserve, max_reserve));

        // Fill with -NaN to deterct incorrect segment write for debugging.
        CHECK_CUDA_ERR(cudaMemset(dev_reserve, 0xff, max_reserve));
    }

    q_seq_array =
        (int*)calloc(main_cfg.batch_size * main_cfg.beam_size, sizeof(int));
    k_seq_array = (int*)calloc(main_cfg.batch_size, sizeof(int));

    if (lo_win_idx == NULL && hi_win_idx == NULL) {
        lo_win_idx = (int*)calloc(main_cfg.seq_len_q, sizeof(int));
        hi_win_idx = (int*)calloc(main_cfg.seq_len_q, sizeof(int));
    }

    // Allocate weight and data buffers on the CPU side.
    if (max_weights > 0) {
        host_w = (T_ELEM*)malloc(max_weights);
    }

    host_q = (T_ELEM*)malloc(max_elem_q * sizeof(T_ELEM));
    host_k = (T_ELEM*)malloc(max_elem_k * sizeof(T_ELEM));
    host_v = (T_ELEM*)malloc(max_elem_v * sizeof(T_ELEM));
    host_o = (T_ELEM*)malloc(max_elem_o * sizeof(T_ELEM));
}

template <typename T_ELEM, typename T_MATH>
void MultiheadAttentionTest<T_ELEM, T_MATH>::Run() {
    AttnConfig test_cfg;

    TestGen(&test_cfg);

    CHECK_CUDNN_ERR(
        cudnnSetDropoutDescriptor(drop_desc, handle, test_cfg.dropout_rate,
                                  dropout_buf, dropout_buf_size, 0));

    // Set attention descriptor according to generated test_cfg.
    CHECK_CUDNN_ERR(cudnnSetAttnDescriptor(
        attn_desc, test_cfg.query_map, test_cfg.num_heads, test_cfg.sm_scaler,
        test_cfg.data_type, test_cfg.comp_prec, CUDNN_DEFAULT_MATH, drop_desc,
        NULL, test_cfg.q_size, test_cfg.k_size, test_cfg.v_size,
        test_cfg.q_proj_size, test_cfg.k_proj_size, test_cfg.v_proj_size,
        test_cfg.o_proj_size, test_cfg.seq_len_q, test_cfg.seq_len_k,
        test_cfg.batch_size, test_cfg.beam_size));

    size_t size_weights = 0, size_wk_space = 0, size_reserve = 0;

    CHECK_CUDNN_ERR(cudnnGetMultiHeadAttnBuffers(
        handle, attn_desc, &size_weights, &size_wk_space, NULL));

    // Sanity check so we do not over-run the allocated buffers.
    if (size_weights > max_weights || size_wk_space > max_wk_space ||
        size_reserve > max_reserve) {
        fprintf(stderr,
                "ERROR: cudnnGetMultiHeadAttnBuffers() reported inconsistent "
                "buffer sizes\n\n");
        exit(-1);
    }

    int q_seq_array_size = test_cfg.beam_size * test_cfg.batch_size;
    int k_seq_array_size = test_cfg.batch_size;

    // host-to-device copies
    size_t size = sizeof(q_seq_array[0]) * q_seq_array_size;
    CHECK_CUDA_ERR(cudaMalloc((void**)&dev_q_seq_array, size));
    CHECK_CUDA_ERR(
        cudaMemcpy(dev_q_seq_array, q_seq_array, size, cudaMemcpyHostToDevice));

    size = sizeof(k_seq_array[0]) * k_seq_array_size;
    CHECK_CUDA_ERR(cudaMalloc((void**)&dev_k_seq_array, size));
    CHECK_CUDA_ERR(
        cudaMemcpy(dev_k_seq_array, k_seq_array, size, cudaMemcpyHostToDevice));

    // Length of output vectors.
    int o_len = test_cfg.o_length();

    int dim_a[CUDNN_SEQDATA_DIM_COUNT];

    dim_a[CUDNN_SEQDATA_BEAM_DIM] = test_cfg.beam_size;
    dim_a[CUDNN_SEQDATA_BATCH_DIM] = test_cfg.batch_size;
    dim_a[CUDNN_SEQDATA_TIME_DIM] = test_cfg.seq_len_q;
    dim_a[CUDNN_SEQDATA_VECT_DIM] = test_cfg.q_size;

    CHECK_CUDNN_ERR(cudnnSetSeqDataDescriptor(
        q_desc, test_cfg.data_type, CUDNN_SEQDATA_DIM_COUNT, dim_a,
        test_cfg.data_axes, q_seq_array_size, q_seq_array, NULL));

    dim_a[CUDNN_SEQDATA_BEAM_DIM] = test_cfg.beam_size;
    dim_a[CUDNN_SEQDATA_BATCH_DIM] = test_cfg.batch_size;
    dim_a[CUDNN_SEQDATA_TIME_DIM] = test_cfg.seq_len_q;
    dim_a[CUDNN_SEQDATA_VECT_DIM] = o_len;

    CHECK_CUDNN_ERR(cudnnSetSeqDataDescriptor(
        o_desc, test_cfg.data_type, CUDNN_SEQDATA_DIM_COUNT, dim_a,
        test_cfg.data_axes, q_seq_array_size, q_seq_array, NULL));

    // seq-k
    dim_a[CUDNN_SEQDATA_BEAM_DIM] =
        test_cfg.query_map == CUDNN_ATTN_QUERYMAP_ONE_TO_ONE
            ? test_cfg.beam_size
            : 1;
    dim_a[CUDNN_SEQDATA_BATCH_DIM] = test_cfg.batch_size;
    dim_a[CUDNN_SEQDATA_TIME_DIM] = test_cfg.seq_len_k;
    dim_a[CUDNN_SEQDATA_VECT_DIM] = test_cfg.k_size;

    CHECK_CUDNN_ERR(cudnnSetSeqDataDescriptor(
        k_desc, test_cfg.data_type, CUDNN_SEQDATA_DIM_COUNT, dim_a,
        test_cfg.data_axes, k_seq_array_size, k_seq_array, NULL));

    // seq-v
    dim_a[CUDNN_SEQDATA_BEAM_DIM] =
        test_cfg.query_map == CUDNN_ATTN_QUERYMAP_ONE_TO_ONE
            ? test_cfg.beam_size
            : 1;
    dim_a[CUDNN_SEQDATA_BATCH_DIM] = test_cfg.batch_size;
    dim_a[CUDNN_SEQDATA_TIME_DIM] = test_cfg.seq_len_k;
    dim_a[CUDNN_SEQDATA_VECT_DIM] = test_cfg.v_size;

    CHECK_CUDNN_ERR(cudnnSetSeqDataDescriptor(
        v_desc, test_cfg.data_type, CUDNN_SEQDATA_DIM_COUNT, dim_a,
        test_cfg.data_axes, k_seq_array_size, k_seq_array, NULL));

    size_t q_num_elem = test_cfg.q_all_data();
    size_t k_num_elem = test_cfg.k_all_data();
    size_t v_num_elem = test_cfg.v_all_data();
    size_t o_nmb_elem = test_cfg.o_all_data();

    size_t q_nmb_weights = test_cfg.q_all_weights();
    size_t k_nmb_weights = test_cfg.k_all_weights();
    size_t v_nmb_weights = test_cfg.v_all_weights();
    size_t o_nmb_weights = test_cfg.o_all_weights();

    // Sanity check so we do not over-run the allocated buffers.
    if (q_num_elem > max_elem_q || k_num_elem > max_elem_k ||
        v_num_elem > max_elem_v || o_nmb_elem > max_elem_o) {
        fprintf(stderr, "ERROR: inconsistent data buffer sizes\n\n");
        exit(-1);
    }

    if (q_num_elem == 0 || k_num_elem == 0 || o_nmb_elem == 0) {
        fprintf(stderr, "ERROR: Q/K/O data buffers cannot be zero size\n\n");
        exit(-1);
    }

    if (size_weights > 0) {
        InitBuffer<T_ELEM>(host_w, size_weights / sizeof(T_ELEM), INIT_MEAN,
                           INIT_VAR);
    }

    InitBuffer<T_ELEM>(host_q, q_num_elem, INIT_MEAN, INIT_VAR);
    InitBuffer<T_ELEM>(host_k, k_num_elem, INIT_MEAN, INIT_VAR);
    InitBuffer<T_ELEM>(host_v, v_num_elem, INIT_MEAN, INIT_VAR);

    // Fill output surface with NaN-s.
    CHECK_CUDA_ERR(cudaMemset(dev_o, 0xFF, o_nmb_elem * sizeof(dev_o[0])));

    // Copy the data from GPU (device) to CPU (host)
    CHECK_CUDA_ERR(
        cudaMemcpy(dev_w, host_w, size_weights, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(dev_q, host_q, sizeof(dev_q[0]) * q_num_elem,
                              cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(dev_k, host_k, sizeof(dev_k[0]) * k_num_elem,
                              cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(dev_v, host_v, sizeof(dev_v[0]) * v_num_elem,
                              cudaMemcpyHostToDevice));

    if (size_reserve != 0) {
        fprintf(stderr,
                "ERROR: non-zero reserve buffer size in inference mode\n\n");
        exit(-1);
    }

    for (int i = 0; i < 5; ++i)
        CHECK_CUDNN_ERR(cudnnMultiHeadAttnForward(
            handle, attn_desc, -1 /*All q time steps are availiable*/,
            lo_win_idx, hi_win_idx, dev_q_seq_array, dev_k_seq_array, q_desc,
            dev_q, main_cfg.res_link ? dev_q : NULL, k_desc, dev_k, v_desc,
            dev_v, o_desc, dev_o, size_weights, size_weights > 0 ? dev_w : NULL,
            size_wk_space, dev_wk_space, 0 /*reserveSpaceSizeInBytes*/,
            NULL /*reserveSpace*/));

    CHECK_CUDA_ERR(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int ITERS = 10;
    cudaEventRecord(start, 0);
    float elapsed = 0.;
    for (int i = 0; i < ITERS; ++i) {
        CHECK_CUDNN_ERR(cudnnMultiHeadAttnForward(
            handle, attn_desc, -1, lo_win_idx, hi_win_idx, dev_q_seq_array,
            dev_k_seq_array, q_desc, dev_q, main_cfg.res_link ? dev_q : NULL,
            k_desc, dev_k, v_desc, dev_v, o_desc, dev_o, size_weights,
            size_weights > 0 ? dev_w : NULL, size_wk_space, dev_wk_space, 0,
            NULL));
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);

    printf("%.*f|\n", 3, elapsed / ITERS);

    // Copy forward output to host.
    CHECK_CUDA_ERR(cudaMemcpy(host_o, dev_o, o_nmb_elem * sizeof(dev_o[0]),
                              cudaMemcpyDeviceToHost));
}

// Teardown destroys various descriptors and free memories.
template <typename T_ELEM, typename T_MATH>
void MultiheadAttentionTest<T_ELEM, T_MATH>::TearDown() {
    cudnnDestroyAttnDescriptor(attn_desc);
    attn_desc = NULL;

    cudnnDestroyDropoutDescriptor(drop_desc);
    drop_desc = NULL;

    cudnnDestroySeqDataDescriptor(q_desc);
    q_desc = NULL;

    cudnnDestroySeqDataDescriptor(k_desc);
    k_desc = NULL;

    cudnnDestroySeqDataDescriptor(v_desc);
    v_desc = NULL;

    cudnnDestroySeqDataDescriptor(o_desc);
    o_desc = NULL;

    cudaFree(dropout_buf);
    dropout_buf = NULL;

    cudaFree(dev_q);
    dev_q = NULL;

    cudaFree(dev_k);
    dev_k = NULL;

    cudaFree(dev_v);
    dev_v = NULL;

    cudaFree(dev_o);
    dev_o = NULL;

    cudaFree(dev_w);
    dev_w = NULL;

    cudaFree(dev_wk_space);
    dev_wk_space = NULL;

    cudaFree(dev_reserve);
    dev_reserve = NULL;

    free(q_seq_array);
    q_seq_array = NULL;

    free(k_seq_array);
    k_seq_array = NULL;

    free(lo_win_idx);
    lo_win_idx = NULL;

    free(hi_win_idx);
    hi_win_idx = NULL;

    free(host_w);
    host_w = NULL;

    free(host_q);
    host_q = NULL;

    free(host_k);
    host_k = NULL;

    free(host_v);
    host_v = NULL;

    free(host_o);
    host_o = NULL;
}

template <typename T_ELEM, typename T_MATH>
void MultiheadAttentionTest<T_ELEM, T_MATH>::TestGen(AttnConfig* test_cfg,
                                                     bool debug_info) {
    *test_cfg = this->main_cfg;

    // Initialize q_seq_array and k_seq_array values and attention window
    size_t q_batches = test_cfg->q_seq_len_count();
    size_t k_batches = test_cfg->k_seq_len_count();

    // Set random number generator seed.
    srand48(test_cfg->rand_seed);

    // No problem size randomization when the RNG seed is zero.
    if (test_cfg->rand_geom != 0) {
        for (size_t i = 0; i < q_batches; ++i) {
            q_seq_array[i] = RandRangeInt(1, test_cfg->seq_len_q);
        }

        for (size_t i = 0; i < k_batches; ++i) {
            k_seq_array[i] = RandRangeInt(1, test_cfg->seq_len_k);
        }

        // Set the random size of attention window in all time-steps.
        for (int i = 0; i < test_cfg->seq_len_q; ++i) {
            lo_win_idx[i] = RandRangeInt(0, test_cfg->seq_len_k - 1);
            hi_win_idx[i] = RandRangeInt(lo_win_idx[i], test_cfg->seq_len_k);
        }
    } else {
        // Fixed lengths for all sequences in a batch.
        for (size_t i = 0; i < q_batches; ++i) {
            q_seq_array[i] = test_cfg->seq_len_q;
        }

        for (size_t i = 0; i < k_batches; ++i) {
            k_seq_array[i] = test_cfg->seq_len_k;
        }

        // Set the maximum attention window in all time-steps.
        for (int i = 0; i < test_cfg->seq_len_q; ++i) {
            lo_win_idx[i] = 0;
            hi_win_idx[i] = test_cfg->seq_len_k;
        }
    }

    const char standard_axes[CUDNN_SEQDATA_DIM_COUNT] = {'T', 'N', 'B', 'V'};
    char data_axes[CUDNN_SEQDATA_DIM_COUNT];
    for (int ii = 0; ii < CUDNN_SEQDATA_DIM_COUNT; ++ii) {
        data_axes[ii] = standard_axes[test_cfg->data_axes[ii]];
    }

    if (debug_info) {
        printf("Test parameters:\n\n");
        printf("#### attnDataType    = %d (FP%d)\n", test_cfg->data_type,
               int(8 * sizeof(T_ELEM)));
        printf("#### attnNumHeads    = %d\n", test_cfg->num_heads);
        printf("#### attnBatchSize   = %d\n", test_cfg->batch_size);
        printf("#### attnBeamSize    = %d\n", test_cfg->beam_size);
        printf("#### attnSmScaler    = %.4e\n", test_cfg->sm_scaler);
        printf("#### attnDropoutRate = %.4f\n", test_cfg->dropout_rate);
        printf("#### attnQsize       = %d\n", test_cfg->q_size);
        printf("#### attnKsize       = %d\n", test_cfg->k_size);
        printf("#### attnVsize       = %d\n", test_cfg->v_size);
        printf("#### attnProjQsize   = %d%s\n", test_cfg->q_proj_size,
               test_cfg->q_proj_size ? "" : " (no Q weights)");
        printf("#### attnProjKsize   = %d%s\n", test_cfg->k_proj_size,
               test_cfg->k_proj_size ? "" : " (no K weights)");
        printf("#### attnProjVsize   = %d%s\n", test_cfg->v_proj_size,
               test_cfg->v_proj_size ? "" : " (no V weights)");
        printf("#### attnProjOsize   = %d%s\n", test_cfg->o_proj_size,
               test_cfg->o_proj_size ? "" : " (no O weights)");
        printf("#### attnSeqLenQ     = %d\n", test_cfg->seq_len_q);
        printf("#### attnSeqLenK     = %d\n", test_cfg->seq_len_k);
        printf("#### attn_data_layout  = %d (%c,%c,%c,%c)\n",
               test_cfg->data_layout, data_axes[0], data_axes[1], data_axes[2],
               data_axes[3]);
        printf("#### attnResLink     = %d\n", test_cfg->res_link);
        printf("#### attnSweep       = %d\n", test_cfg->sweep);
        printf("#### attnRandGeom    = %d\n", test_cfg->rand_geom);
        printf("#### attnRandSeed    = %d\n", test_cfg->rand_seed);

        for (size_t i = 0; i < q_batches; ++i) {
            printf("sequence_length_q[idx=%lu]=%d\n", i, q_seq_array[i]);
        }
        printf("\n");

        for (size_t i = 0; i < k_batches; ++i) {
            printf("sequence_length_k[idx=%lu]=%d\n", i, k_seq_array[i]);
        }
        printf("\n");

        for (int i = 0; i < test_cfg->seq_len_q; ++i) {
            printf("attention_window[time=%d]=%d:%d\n", i, lo_win_idx[i],
                   hi_win_idx[i]);
        }
        printf("\n");
    }
}
