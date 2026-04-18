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
 * @FilePath: /Artea/include/artea/cpu/router/detail/make_layer_range.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Picks the right NeighborRange adapter for a hierarchical
 *               graph at a given level. Dispatch is on the graph's
 *               @c is_compacted static flag (compact graphs yield raw
 *               vertex_id_t; dynamic graphs yield nbr_t and need
 *               vid projection).
 */

#pragma once

#include <artea/cpu/router/detail/compact_layer_range.hpp>
#include <artea/cpu/router/detail/dynamic_layer_range.hpp>

namespace artea {
namespace cpu {
namespace detail {

template <typename HierarchicalGraphT>
__attribute__((always_inline))
inline auto make_layer_range(
    const HierarchicalGraphT& hier_graph,
    const typename HierarchicalGraphT::layer_id_t level_id
) {
    if constexpr (HierarchicalGraphT::is_compacted) {
        return CompactLayerRange<HierarchicalGraphT>(hier_graph, level_id);
    } else {
        return DynamicLayerRange<HierarchicalGraphT>(hier_graph, level_id);
    }
}

}   // namespace detail
}   // namespace cpu
}   // namespace artea
