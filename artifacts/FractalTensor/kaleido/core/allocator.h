#pragma once

#include <cstddef>

namespace kaleido {
namespace core {

// TODO(ying): make Place a template parameter.
// template <typename Place>
class Allocator {
   public:
    virtual ~Allocator() = default;

    virtual void* Allocate(const size_t& nbytes) = 0;
    virtual void Deallocate(void* ptr) = 0;
};
}  // namespace core
}  // namespace kaleido
