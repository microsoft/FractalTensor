#pragma once

#include "boost/variant.hpp"

namespace kaleido {
namespace core {

struct CPUPlace {
    CPUPlace() {}

    inline bool operator==(const CPUPlace&) const { return true; }
    inline bool operator!=(const CPUPlace&) const { return false; }
    inline bool operator<(const CPUPlace&) const { return false; }
};

struct CUDAPlace {
    CUDAPlace() : CUDAPlace(0) {}
    explicit CUDAPlace(int d) : device(d) {}

    inline int GetDeviceId() const { return device; }
    inline bool operator==(const CUDAPlace& o) const {
        return device == o.device;
    }
    inline bool operator!=(const CUDAPlace& o) const { return !(*this == o); }
    inline bool operator<(const CUDAPlace& o) const {
        return device < o.device;
    }

    int device;
};

class Place : public boost::variant<CUDAPlace, CPUPlace> {
   private:
    using PlaceBase = boost::variant<CUDAPlace, CPUPlace>;

   public:
    Place() = default;
    Place(const CPUPlace& cpu_place) : PlaceBase(cpu_place) {}
    Place(const CUDAPlace& cuda_place) : PlaceBase(cuda_place) {}

    bool operator<(const Place& place) const {
        return PlaceBase::operator<(static_cast<const PlaceBase&>(place));
    }

    bool operator==(const Place& place) const {
        return PlaceBase::operator==(static_cast<const PlaceBase&>(place));
    }
};

std::ostream& operator<<(std::ostream&, const Place&);

}  // namespace core
}  // namespace kaleido
