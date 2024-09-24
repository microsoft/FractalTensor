#pragma once
#include "kaleido/core/allocator.h"
#include "kaleido/core/types.h"

namespace kaleido {
namespace core {

class FractalTensor {
   public:
    FractalTensor(const FractalTensorTypeDesc& desc,
                  std::shared_ptr<Allocator> alloc)
        : type_desc_(desc), alloc_(alloc), data_(nullptr) {
        long byteCount = type_desc_.GetNumBytes();
        if (byteCount) data_ = alloc_->Allocate(byteCount);
    };

    ~FractalTensor() = default;

    std::string DebugString() const;

    template <typename T>
    const T* data() const {
        return reinterpret_cast<T*>(data_);
    }

    template <typename T>
    T* mutable_data() {
        return reinterpret_cast<T*>(data_);
    }

   private:
    FractalTensorTypeDesc type_desc_;
    std::shared_ptr<Allocator> alloc_;

    void* data_;
};
static inline std::ostream& operator<<(std::ostream& os,
                                       const FractalTensor& ft) {
    os << ft.DebugString();
    return os;
}

}  // namespace core
}  // namespace kaleido
