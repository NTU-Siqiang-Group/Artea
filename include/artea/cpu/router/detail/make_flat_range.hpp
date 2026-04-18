// Copyright 2026 Weitang Ye
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*
 * @FilePath: /Artea/include/artea/cpu/router/detail/make_flat_range.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Picks the right NeighborRange adapter for a flat
 *               (refining) graph. Dispatches on the graph's
 *               @c is_compacted static flag.
 */

#pragma once

#include <artea/cpu/router/detail/compact_flat_range.hpp>
#include <artea/cpu/router/detail/dynamic_flat_range.hpp>

namespace artea {
namespace cpu {
namespace detail {

template <typename SingleLayerGraphT>
__attribute__((always_inline))
inline auto make_flat_range(const SingleLayerGraphT& single_layer_graph) {
    if constexpr (SingleLayerGraphT::is_compacted) {
        return CompactFlatRange<SingleLayerGraphT>(single_layer_graph);
    } else {
        return DynamicFlatRange<SingleLayerGraphT>(single_layer_graph);
    }
}

}   // namespace detail
}   // namespace cpu
}   // namespace artea
