#include "multi_head_attention.h"

int main(int argc, char** argv) {
    TestOpts opts;
    ParseAttnParameters(argc, argv, &opts);

    MultiheadAttentionTest<float, float> attn_test;
    attn_test.SetUp(opts);
    attn_test.Run();
    attn_test.TearDown();

    fflush(stdout);

    return 0;
}
