#include "kaleido/core/cuda_allocator.h"
#include "kaleido/core/layout.h"

#include <gtest/gtest.h>

#include <iostream>

namespace kaleido {
namespace core {

TEST(test, TestLayout) {
    const int kRow = 3;
    const int kCol = 7;
    using L1 = RowMajor<kRow, kCol>;
    L1 row_major;

    std::cout << "num_rows: " << num_rows<L1> << std::endl
              << "num_cols: " << num_cols<L1>;

    for (int row_id = 0; row_id < num_rows<L1>; ++row_id) {
        for (int col_id = 0; col_id < num_cols<L1>; ++col_id) {
            EXPECT_EQ(row_major(row_id, col_id), row_id * kCol + col_id);
        }
    }

    using L2 = ColMajor<kRow, kCol>;
    L2 col_major;
    for (int row_id = 0; row_id < num_rows<L2>; ++row_id) {
        for (int col_id = 0; col_id < num_rows<L2>; ++col_id) {
            EXPECT_EQ(col_major(row_id, col_id), row_id + col_id * kRow);
        }
    }
}

}  // namespace core
}  // namespace kaleido
