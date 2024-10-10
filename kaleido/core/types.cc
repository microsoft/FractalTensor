// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "kaleido/core/types.h"

#include <glog/logging.h>

namespace kaleido {
namespace core {

const std::string DataType::TypeToString() const {
    switch (type_) {
        case PrimitiveType::FLOAT32:
            return "float32";
        case PrimitiveType::FLOAT64:
            return "float64";
        case PrimitiveType::INT32:
            return "int32";
        case PrimitiveType::INT64:
            return "int64";
        case PrimitiveType::UINT32:
            return "unsigned int32";
        case PrimitiveType::UINT64:
            return "unsigned int64";
        case PrimitiveType::BOOL:
            return "bool";
        default:
            LOG(FATAL) << "Unknown type.";
    }
}

int64_t DataType::ByteSizeOfType() const {
    switch (type_) {
        case PrimitiveType::FLOAT32:
            return sizeof(float);
        case PrimitiveType::FLOAT64:
            return sizeof(double);
        case PrimitiveType::INT32:
            return sizeof(int32_t);
        case PrimitiveType::INT64:
            return sizeof(int64_t);
        case PrimitiveType::UINT32:
            return sizeof(uint32_t);
        case PrimitiveType::UINT64:
            return sizeof(uint64_t);
        case PrimitiveType::BOOL:
            return sizeof(bool);
        default:
            LOG(FATAL) << "Unknown type.";
    }
}

bool DataType::operator<(const DataType& other) const {
    LOG(FATAL) << "Not implemented yet.";
}

bool DataType::operator==(const DataType& other) const {
    LOG(FATAL) << "Not implemented yet.";
}

const DataType TensorTypeDesc::StringToType(
    const std::string& type_name) const {
    if (type_name == "float32")
        return DataType(PrimitiveType::FLOAT32);
    else if (type_name == "float64")
        return DataType(PrimitiveType::FLOAT64);
    else if (type_name == "int32")
        return DataType(PrimitiveType::INT32);
    else if (type_name == "int64")
        return DataType(PrimitiveType::INT64);
    else if (type_name == "uint32")
        return DataType(PrimitiveType::UINT32);
    else if (type_name == "uint64")
        return DataType(PrimitiveType::UINT64);
    else if (type_name == "bool")
        return DataType(PrimitiveType::BOOL);
    else
        LOG(FATAL) << "Unknown type: " << type_name << std::endl;
}

Layout TensorTypeDesc::StringToLayout(const std::string& layout_name) const {
    if (layout_name == "col")
        return Layout::COL_MAJOR;
    else if (layout_name == "row")
        return Layout::ROW_MAJOR;
    else
        LOG(FATAL) << "Unknown layout: " << layout_name << std::endl;
}

long long int TensorTypeDesc::GetNumBytes() const {
    return shape_.numel_ * dtype_.ByteSizeOfType();
}

std::string FractalTensorTypeDesc::PrintIndexTree() const {
    size_t depth = is_static_.size();

    if (indices_.size() && indices_.size() != is_static_.size())
        LOG(FATAL) << "the length of indices_ and the length of static_flags "
                      "shoule be the same.";

    std::stringstream ss;
    ss << "[" << std::endl;
    for (size_t i = 0; i < depth; ++i) {
        ss << "      [";
        if (indices_.size()) {
            for (size_t j = 0; j < indices_[i].size() - 1; ++j)
                ss << indices_[i][j] << ", ";
            ss << indices_[i][indices_[i].size() - 1];
        } else {
            ss << "  -- uninitialized --  ";
        }
        ss << "]    // depth-" << depth - i
           << (is_static_[i] ? "-static" : "-dynamic") << std::endl;
    }
    ss << "  ]";
    return ss.str();
}

std::string FractalTensorTypeDesc::DebugString() const {
    std::stringstream ss;
    ss << "FractalTensorTypeDesc {" << std::endl
       << dtype_.DebugString() << std::endl
       << "  depth: " << depth_ << std::endl
       << "  indices: " << PrintIndexTree() << std::endl
       << "}" << std::endl;
    return ss.str();
}

size_t FractalTensorTypeDesc::GetElementCount() const {
    int n = indices_[depth_ - 1].size();
    return indices_[depth_ - 1][n - 1];
}

long long int FractalTensorTypeDesc::GetNumBytes() const {
    if (indices_.empty()) return 0;

    return GetElementCount() * dtype_.GetNumBytes();
}

}  // namespace core
}  // namespace kaleido
