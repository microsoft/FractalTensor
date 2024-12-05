#pragma once
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cutlass/numeric_conversion.h"

namespace cute {

template <typename To_type, typename Engine, typename Layout>
CUTE_DEVICE auto convert_type(cute::Tensor<Engine, Layout> const& tensor) {
    using From_type = typename Engine::value_type;
    constexpr int numel = decltype(size(tensor))::value;
    cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
    auto frag =
        convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel>*>(
            tensor.data()));
    return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}

template <typename Layout>
inline __device__ auto convert_layout_rowcol_Aregs(Layout rowcol_layout) {
    using namespace cute;
    static_assert(decltype(size<0, 0>(rowcol_layout))::value == 2);
    static_assert(decltype(size<1, 0>(rowcol_layout))::value == 2);
    auto l = logical_divide(rowcol_layout,
                            Shape<Underscore, Shape<Underscore, Int<2>>>{});

    return make_layout(make_layout(get<0>(get<1>(l)), get<0>(get<0>(l)),
                                   get<0>(get<1>(get<1>(l)))),
                       get<1>(get<0>(l)), get<1>(get<1>(get<1>(l))));
}

inline __device__ auto convert_layout_C_Aregs() {
    using namespace cute;
    auto layout_s = Layout<Shape<Shape<_2, _2>, _2, _16>>{};
    auto l = logical_divide(layout_s, Shape<Underscore, Underscore, _2>{});

    return make_layout(
        make_layout(get<0>(get<0>(l)), get<1>(get<0>(l)), get<0>(get<2>(l))),
        get<1>(l), get<1>(get<2>(l)));
}

template <class LayoutType>
inline __device__ auto convert_layout_scores(LayoutType layout_s) {
    using namespace cute;
    static_assert(decltype(size<0>(layout_s))::value == 4);
    static_assert(decltype(rank(layout_s))::value == 3);

    auto l = logical_divide(layout_s, Shape<_2>{});
    return make_layout(make_layout(get<1>(get<0>(l)), get<1>(l)),
                       make_layout(get<0>(get<0>(l)), get<2>(l)));
}

template <int ATOMNUM, class LayoutType>
inline __device__ auto convert_layout_scores_copyview(LayoutType layout_s) {
    using namespace cute;

    auto l = logical_divide(layout_s, Shape<Underscore, Int<ATOMNUM>>{});
    return make_layout(get<0>(get<1>(l)), get<0>(l), get<1>(get<1>(l)));
}
}  // namespace cute