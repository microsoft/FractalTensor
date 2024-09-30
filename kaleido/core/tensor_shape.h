#pragma once

#include <glog/logging.h>

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace kaleido {
namespace core {

// The tensor shape is static after declaration.
class TensorShape {
   public:
    explicit TensorShape(std::initializer_list<int64_t> sizes)
        : dim_sizes_(sizes),
          dim_(sizes.size()),
          numel_(std::accumulate(sizes.begin(), sizes.end(), 1,
                                 std::multiplies<int64_t>())) {}
    explicit TensorShape(const std::vector<int64_t> sizes)
        : dim_sizes_(std::move(sizes)),
          dim_(sizes.size()),
          numel_(std::accumulate(sizes.begin(), sizes.end(), 1,
                                 std::multiplies<int64_t>())) {}
    ~TensorShape() = default;
    bool IsEuqalShape(const TensorShape& b) const;
    void operator=(TensorShape& b) {
        dim_sizes_ = std::move(b.dims());
        dim_ = b.ndim();
        numel_ = b.numel();
    };

    bool operator==(const TensorShape& b) const { return IsEuqalShape(b); };
    bool operator!=(const TensorShape& b) const { return !IsEuqalShape(b); };

    std::string DebugString() const;

    size_t ndim() const { return dim_; }
    int64_t dim_size(int i) const {
        return i >= 0 ? dim_sizes_[i] : dim_sizes_[dim_ + i];
    }

    const std::vector<int64_t>& dims() const { return dim_sizes_; }
    int64_t numel() const { return numel_; }

    int64_t count(int i) const {
        return std::accumulate(dim_sizes_.begin() + i, dim_sizes_.end(), 1,
                               std::multiplies<int64_t>());
    }

    std::vector<int64_t> dim_sizes_;
    size_t dim_;
    int64_t numel_;
};

}  // namespace core
}  // namespace kaleido
