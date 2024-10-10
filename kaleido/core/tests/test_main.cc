// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <glog/logging.h>
#include <gtest/gtest.h>

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    google::InitGoogleLogging(argv[0]);

    return RUN_ALL_TESTS();
}
