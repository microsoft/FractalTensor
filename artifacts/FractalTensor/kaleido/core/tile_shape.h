#pragma once

#include "kaleido/core/config.h"

#include <cute/container/tuple.hpp>
#include <cute/int_tuple.hpp>

namespace kaleido {
namespace core {

template <size_t... T>
using TileShape = cute::tuple<std::integral_constant<size_t, T>...>;

// FIXME(ying): Be careful that names like `rank` is quite common.
// It is easy to conflict with cute's builtin function.
template <typename TileShape>
__device__ static constexpr size_t rank = cute::rank_v<TileShape>;

template <const size_t I, typename TileShape>
__device__ static constexpr size_t dim_size = cute::get<I>(TileShape{});

template <typename TileShape>
__device__ static constexpr int64_t get_numel = cute::size(TileShape{});

}  // namespace core
}  // namespace kaleido
