// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <glog/logging.h>
#include <pybind11/pybind11.h>

#include <mutex>

namespace py = pybind11;

namespace kaleido {
namespace core {

std::once_flag glog_init_flag;

void InitGLOG(const std::string& prog_name) {
    std::call_once(glog_init_flag, [&]() {
        google::InitGoogleLogging(strdup(prog_name.c_str()));
    });
}

PYBIND11_MODULE(_core, m) { m.def("init_glog", InitGLOG); }

}  // namespace core
}  // namespace kaleido
