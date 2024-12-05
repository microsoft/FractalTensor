#include "kaleido/core/cuda_allocator.h"
#include "kaleido/core/device/cuda_utils.h"

#include <gtest/gtest.h>

#include <iostream>
#include <mutex>
#include <thread>

namespace kaleido {
namespace core {

TEST(test1, TEST_MEMORY_POOL) {
    cudaStream_t stream;
    CudaCheck(cudaStreamCreate(&stream));

    std::shared_ptr<CudaMemoryPool> memoryPool =
        std::make_shared<CudaMemoryPool>();
    // CudaMemoryPool is NOT multi-thread safe,
    // prevent multiple threads from modifying the memory pool at the same time.
    std::mutex mtx;

    // Before allocating memory from the memory pool,
    // you need to register **all** the streams that may use memory space
    // allocated from the memory pool.
    const std::lock_guard<std::mutex> lock(mtx);
    memoryPool->add_track_stream(stream);

    // Get 256MB cuda memory block.
    // Only when the memory pool does not have a memory block that meets the
    // requirements, a new memory block is actually allocated from the physical
    // device. Requirements:
    // - The returned memory block size should be greater than the required
    // size.
    // - The returned memory block size should be less than twice the requested
    // size.

    // The returned memory space is guaranteed to meet the requested size.
    // If the user reads and writes beyond the requested size, undefined
    // behavior may occur.
    int nbytes = 256 * 1024 * 2014;
    void* ret = memoryPool->Allocate(nbytes);

    // Put the memory space back into the memory pool.
    memoryPool->Deallocate(ret);

    nbytes = 128 * 1024 * 2014;
    ret = memoryPool->Allocate(nbytes);
    memoryPool->Deallocate(ret);

    nbytes = 128 * 1024 * 2014;
    ret = memoryPool->Allocate(nbytes);
    memoryPool->Deallocate(ret);
}

}  // namespace core
}  // namespace kaleido
