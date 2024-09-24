#pragma once

#include "kaleido/core/allocator.h"
#include "kaleido/core/types.h"

namespace kaleido {
namespace core {

class Tensor {
   public:
    Tensor(const TensorTypeDesc& desc, std::shared_ptr<Allocator> alloc)
        : type_desc_(desc), alloc_(alloc) {
        data_ = alloc_->Allocate(type_desc_.GetNumBytes());
    }

    explicit Tensor(std::initializer_list<int64_t> sizes,
                    std::shared_ptr<Allocator> alloc,
                    const std::string& dtype = "float32",
                    const std::string device = "cuda",
                    const std::string& layout = "row")
        : type_desc_(TensorTypeDesc(sizes, dtype, device, layout)) {
        if (device != "cuda") {
            LOG(FATAL) << "Not implemented. " << std::endl;
        }

        alloc_ = alloc;
        if (alloc_) {
            data_ = alloc_->Allocate(type_desc_.GetNumBytes());
        } else {
            data_ = nullptr;
        }
    }
    ~Tensor() = default;
    std::string DebugString() const;

    template <typename T>
    void CreateFrom(const Tensor& a, long long int offset) {
        // TODO(ying): is it necessary to support the concept like
        // `View`.
        if (alloc_) LOG(FATAL) << "alloc_ should be nullptr.";

        CHECK_LE(offset + numel(), a.numel()) << "Out of boundary.";
        data_ = reinterpret_cast<void*>(
            reinterpret_cast<T*>(a.GetDataPoiner()) + offset);
    }

    template <typename T>
    static std::shared_ptr<Tensor> CreateView(
        const Tensor& source, std::initializer_list<int64_t> sizes,
        long long int offset) {
        auto target = std::make_shared<Tensor>(sizes, nullptr);
        target->CreateFrom<float>(source, offset);
        return target;
    }

    template <typename T>
    static std::shared_ptr<Tensor> ReshapeFrom(
        const Tensor& source, std::initializer_list<int64_t> sizes) {
        int64_t new_numel = std::accumulate(sizes.begin(), sizes.end(), 1,
                                            std::multiplies<int64_t>());
        CHECK_EQ(new_numel, source.numel());

        auto target = std::make_shared<Tensor>(sizes, nullptr);
        target->CreateFrom<float>(source, 0);
        return target;
    }

    template <typename T>
    const T* data() const {
        return reinterpret_cast<T*>(data_);
    }

    template <typename T>
    T* mutable_data() {
        return reinterpret_cast<T*>(data_);
    }

    void* GetDataPoiner() const { return data_; }

    int64_t numel() const { return type_desc_.numel(); }
    int64_t ndim() const { return type_desc_.shape().ndim(); }
    int64_t dim_size(int i) const { return type_desc_.shape().dim_size(i); }
    const std::vector<int64_t>& dims() const {
        return type_desc_.shape().dims();
    }
    Layout layout() const { return type_desc_.layout(); }
    const DataType dtype() const { return type_desc_.dtype(); }

    const TensorShape shape() const { return type_desc_.shape(); }

   private:
    TensorTypeDesc type_desc_;
    std::shared_ptr<Allocator> alloc_;

    void* data_;
};
static inline std::ostream& operator<<(std::ostream& os, const Tensor& t) {
    os << t.DebugString();
    return os;
}

}  // namespace core
}  // namespace kaleido
