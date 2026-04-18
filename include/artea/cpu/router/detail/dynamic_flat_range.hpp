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
 * @FilePath: /Artea/include/artea/cpu/router/detail/dynamic_flat_range.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: NeighborRange adapter binding a dynamic::RefiningGraph.
 *               Projects nbr_t -> vertex_id_t and sentinel-stops on
 *               nbr.is_invalid() (or the storage end, whichever is first).
 */

#pragma once

#include <ranges>

namespace artea {
namespace cpu {
namespace detail {

template <typename RefiningGraphT>
class DynamicFlatRange {
public:
    using vertex_id_t = typename RefiningGraphT::vertex_id_t;
    using nbr_t       = typename RefiningGraphT::nbr_t;

    __attribute__((always_inline))
    explicit DynamicFlatRange(const RefiningGraphT& refining_graph) :
        _refining_graph(refining_graph) {}

    __attribute__((always_inline))
    auto of(const vertex_id_t vid) const {
        return _refining_graph.fetch_nbrs(vid)
             | std::views::take_while([](const nbr_t& nbr) { return !nbr.is_invalid(); })
             | std::views::transform([](const nbr_t& nbr) { return nbr.get_vid(); });
    }

private:
    const RefiningGraphT& _refining_graph;

};  // class DynamicFlatRange

}   // namespace detail
}   // namespace cpu
}   // namespace artea
