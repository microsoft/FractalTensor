#pragma once

#include "kaleido/core/place.h"
#include "kaleido/core/tensor_shape.h"

#include <glog/logging.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>

namespace kaleido {
namespace core {

enum class PrimitiveType : size_t {
    BOOL = 1,
    INT32 = 2,
    INT64 = 3,
    UINT32 = 4,
    UINT64 = 5,
    FLOAT32 = 6,
    FLOAT64 = 7,
};

enum class Layout : size_t {
    ROW_MAJOR = 0,
    COL_MAJOR = 1,
};

class DataType {
   public:
    explicit DataType(const PrimitiveType& type) : type_(type) {}

    const std::string TypeToString() const;
    int64_t ByteSizeOfType() const;

    bool operator<(const DataType& other) const;
    bool operator==(const DataType& other) const;
    bool operator!=(const DataType& other) const { return !(*this == other); }

   private:
    PrimitiveType type_;
    size_t nbytes_;
};
static inline std::ostream& operator<<(std::ostream& os, const DataType& d) {
    os << d.TypeToString();
    return os;
}

class TypeBase {
   public:
    TypeBase() {}
    ~TypeBase() {}

    virtual std::string DebugString() const = 0;
    virtual long long int GetNumBytes() const = 0;
};

class TensorTypeDesc : public TypeBase {
   public:
    explicit TensorTypeDesc(std::vector<int64_t> sizes,
                            const std::string& dtype, const std::string& device,
                            const std::string& layout)
        : shape_(TensorShape(sizes)),
          dtype_(StringToType(dtype)),
          place_(CUDAPlace(0)),
          layout_(StringToLayout(layout)) {
        if (device != "cuda") {
            LOG(FATAL) << "Not implemented. " << std::endl;
        }
    }

    explicit TensorTypeDesc(TensorShape& shape, const PrimitiveType dtype,
                            const Place& place)
        : shape_(shape),
          dtype_(DataType(dtype)),
          place_(place),
          layout_{Layout::ROW_MAJOR} {}  // NOTE: default layout is row-major.

    ~TensorTypeDesc() = default;

    std::string DebugString() const {
        std::stringstream ss;
        ss << "  " << shape_.DebugString() << std::endl
           << "  dtype: " << dtype_.TypeToString() << std::endl
           << "  place: " << place_ << std::endl
           << "  layout: "
           << (static_cast<std::underlying_type<Layout>::type>(layout_) == 0
                   ? "row-major"
                   : "column-major");
        return ss.str();
    }

    int64_t numel() const { return shape_.numel(); }
    const DataType StringToType(const std::string& type_name) const;
    Layout StringToLayout(const std::string& layout_name) const;
    const TensorShape& shape() const { return shape_; }
    long long int GetNumBytes() const;
    Layout layout() const { return layout_; }
    DataType dtype() const { return dtype_; }

   private:
    const TensorShape shape_;
    const DataType dtype_;
    const Place place_;
    const Layout layout_;
};

class FractalTensorTypeDesc : public TypeBase {
   public:
    explicit FractalTensorTypeDesc(const TensorTypeDesc dtype)
        : dtype_(dtype), depth_(1), is_static_({false}), indices_({}) {}
    explicit FractalTensorTypeDesc(const TensorTypeDesc dtype,
                                   std::vector<std::vector<int64_t>> ids,
                                   std::vector<bool> static_flags)
        : dtype_(dtype),
          depth_(ids.size()),
          is_static_(static_flags),
          indices_(ids) {
        if (static_flags.size() != ids.size())
            LOG(FATAL) << "the static flags should have a same length "
                          "as depth.";
    }
    ~FractalTensorTypeDesc() = default;

    std::string DebugString() const;
    std::string PrintIndexTree() const;

    const std::vector<std::vector<int64_t>> GetIndexTree() const {
        return indices_;
    }
    size_t GetElementCount() const;
    long long int GetNumBytes() const;

   private:
    const TensorTypeDesc dtype_;
    const size_t depth_;
    const std::vector<bool> is_static_;

    std::vector<std::vector<int64_t>> indices_;
};

}  // namespace core
}  // namespace kaleido
