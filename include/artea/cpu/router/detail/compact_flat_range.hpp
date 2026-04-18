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
 * @FilePath: /Artea/include/artea/cpu/router/detail/compact_flat_range.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: NeighborRange adapter binding a compact::RefiningGraph.
 *               The graph already yields vertex_id_t directly via
 *               fetch_nbrs(); the adapter just sentinel-stops.
 *               The neighbor cap is baked into compact storage at
 *               compaction time (extracted_nbr_size), so no per-call cap.
 */

#pragma once

#include <ranges>

namespace artea {
namespace cpu {
namespace detail {

template <typename RefiningGraphT>
class CompactFlatRange {
public:
    using vertex_id_t = typename RefiningGraphT::vertex_id_t;

    static constexpr vertex_id_t invalid_vertex_id =
        RefiningGraphT::invalid_vertex_id;

    __attribute__((always_inline))
    explicit CompactFlatRange(const RefiningGraphT& refining_graph) :
        _refining_graph(refining_graph) {}

    __attribute__((always_inline))
    auto of(const vertex_id_t vid) const {
        return _refining_graph.fetch_nbrs(vid)
             | std::views::take_while([](const vertex_id_t v) {
                   return v != invalid_vertex_id;
               });
    }

private:
    const RefiningGraphT& _refining_graph;

};  // class CompactFlatRange

}   // namespace detail
}   // namespace cpu
}   // namespace artea
