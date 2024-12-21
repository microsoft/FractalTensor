// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "kaleido/core/place.h"

#include <iostream>

namespace kaleido {
namespace core {

class PlacePrinter : public boost::static_visitor<> {
public:
  explicit PlacePrinter(std::ostream& os) : os_(os) {}

  void operator()(const CPUPlace&) { os_ << "CPU"; }
  void operator()(const CUDAPlace& p) { os_ << "CUDA:" << p.device; }

private:
  std::ostream& os_;
};

std::ostream& operator<<(std::ostream& out, const Place& place) {
  PlacePrinter printer(out);
  boost::apply_visitor(printer, place);
  return out;
}

}  // namespace core
}  // namespace kaleido
