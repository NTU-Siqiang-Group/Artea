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
 * @FilePath: /Artea/include/artea/cpu/router/detail/compact_layer_range.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: NeighborRange adapter binding a compact::HierarchicalGraph
 *               + layer_id. The graph already yields vertex_id_t directly,
 *               so the adapter just sentinel-stops on invalid_vertex_id.
 */

#pragma once

#include <ranges>

namespace artea {
namespace cpu {
namespace detail {

template <typename HierarchicalGraphT>
class CompactLayerRange {
public:
    using vertex_id_t = typename HierarchicalGraphT::vertex_id_t;
    using layer_id_t  = typename HierarchicalGraphT::layer_id_t;

    static constexpr vertex_id_t invalid_vertex_id =
        HierarchicalGraphT::invalid_vertex_id;

    __attribute__((always_inline))
    CompactLayerRange(const HierarchicalGraphT& hier_graph,
                      const layer_id_t          level_id) :
        _hier_graph(hier_graph), _level_id(level_id) {}

    __attribute__((always_inline))
    auto of(const vertex_id_t vid) const {
        return _hier_graph.fetch_layer_nbrs(vid, _level_id)
             | std::views::take_while([](const vertex_id_t v) {
                   return v != invalid_vertex_id;
               });
    }

private:
    const HierarchicalGraphT& _hier_graph;
    layer_id_t                _level_id;

};  // class CompactLayerRange

}   // namespace detail
}   // namespace cpu
}   // namespace artea
