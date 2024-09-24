#pragma once

#include "kaleido/core/allocator.h"
#include "kaleido/core/device/cuda_utils.h"

#include <glog/logging.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace kaleido {
namespace core {

typedef void* StreamHandle;

class CudaMemoryBlock {
   public:
    explicit CudaMemoryBlock(
        const size_t& nbytes,
        std::shared_ptr<std::unordered_set<StreamHandle>> track_streams)
        : nbytes_(nbytes),
          marked_free_(true),
          track_streams_(std::move(track_streams)) {
        CudaCheck(cudaMalloc((void**)&ptr_, nbytes));
    }

    ~CudaMemoryBlock();

    bool isFree();

    void clearEvents();

    const size_t nbytes_;
    bool marked_free_;
    std::vector<cudaEvent_t> events_;
    const void* ptr_ = {};
    std::shared_ptr<std::unordered_set<StreamHandle>> track_streams_;
};

struct Comp {
    bool operator()(std::shared_ptr<CudaMemoryBlock>& s,
                    const long unsigned int& i) const {
        return s->nbytes_ < i;
    }
};

class CudaMemoryPool : public Allocator {
   public:
    CudaMemoryPool() = default;
    ~CudaMemoryPool() = default;

    void* Allocate(const size_t& min_nbytes);
    void Deallocate(void* ret);
    void add_track_stream(void* stream);

   private:
    std::vector<std::shared_ptr<CudaMemoryBlock>> all_vector_;
    Comp comp_;
    std::unordered_map<const void*, std::shared_ptr<CudaMemoryBlock>> all_map_;
    std::shared_ptr<std::unordered_set<StreamHandle>> track_streams_ =
        std::make_shared<std::unordered_set<StreamHandle>>();
};

CudaMemoryBlock::~CudaMemoryBlock() { CudaCheck(cudaFree((void*)ptr_)); }

bool CudaMemoryBlock::isFree() {
    if (marked_free_) {
        auto result =
            std::all_of(events_.begin(), events_.end(), [](cudaEvent_t e) {
                cudaError_t error = cudaEventQuery(e);
                CHECK(error == cudaSuccess || error == cudaErrorNotReady)
                    << "CUDA: " << cudaGetErrorString(error);
                return error == cudaSuccess;
            });
        if (result) {
            clearEvents();
            return true;
        } else {
            return false;
        }
    } else {
        return false;
    }
}

void CudaMemoryBlock::clearEvents() {
    for (auto& event : events_) {
        CudaCheck(cudaEventDestroy(event));
    }
    events_.clear();
}

void CudaMemoryPool::Deallocate(void* ret) {
    auto find = all_map_.find(ret);
    CHECK(find != all_map_.end());

    CHECK(!(*find).second->marked_free_);

    (*find).second->marked_free_ = true;
}

void* CudaMemoryPool::Allocate(const size_t& min_nbytes) {
    auto lower_bound_iter = std::lower_bound(
        all_vector_.begin(), all_vector_.end(), min_nbytes, comp_);
    if (lower_bound_iter == all_vector_.end()) {
        auto memory_block =
            std::make_shared<CudaMemoryBlock>(min_nbytes, track_streams_);

        CHECK(memory_block->marked_free_);
        CHECK(memory_block->events_.empty());

        memory_block->marked_free_ = false;

        for (auto& stream : *memory_block->track_streams_) {
            cudaEvent_t event;
            CudaCheck(cudaEventCreate(&event));

            auto cuStream = static_cast<cudaStream_t>(stream);
            CudaCheck(cudaEventRecord(event, cuStream));

            memory_block->events_.emplace_back(event);
        }

        auto ptr = (void*)memory_block->ptr_;
        all_map_.insert({ptr, memory_block});
        all_vector_.push_back(memory_block);

        return ptr;
    }

    auto find_if_iter = std::find_if(
        lower_bound_iter, all_vector_.end(),
        [](const std::shared_ptr<CudaMemoryBlock>& s) { return s->isFree(); });

    if (find_if_iter == all_vector_.end() ||
        (*find_if_iter)->nbytes_ / min_nbytes > 2) {
        auto memory_block =
            std::make_shared<CudaMemoryBlock>(min_nbytes, track_streams_);

        CHECK(memory_block->marked_free_);
        CHECK(memory_block->events_.empty());

        memory_block->marked_free_ = false;

        for (auto& stream : *memory_block->track_streams_) {
            cudaEvent_t event;
            CudaCheck(cudaEventCreate(&event));

            auto cuStream = static_cast<cudaStream_t>(stream);
            CudaCheck(cudaEventRecord(event, cuStream));

            memory_block->events_.emplace_back(event);
        }

        auto ptr = (void*)memory_block->ptr_;
        all_map_.insert({ptr, memory_block});
        all_vector_.insert(lower_bound_iter, memory_block);

        return ptr;
    } else {
        CHECK((*find_if_iter)->marked_free_);
        CHECK((*find_if_iter)->events_.empty());

        (*find_if_iter)->marked_free_ = false;

        for (auto& stream : *(*find_if_iter)->track_streams_) {
            cudaEvent_t event;
            CudaCheck(cudaEventCreate(&event));

            auto cuStream = static_cast<cudaStream_t>(stream);
            CudaCheck(cudaEventRecord(event, cuStream));

            (*find_if_iter)->events_.emplace_back(event);
        }

        auto ptr = (void*)(*find_if_iter)->ptr_;
        return ptr;
    }
}

void CudaMemoryPool::add_track_stream(void* stream) {
    track_streams_->emplace(stream);
}

}  // namespace core
}  // namespace kaleido
