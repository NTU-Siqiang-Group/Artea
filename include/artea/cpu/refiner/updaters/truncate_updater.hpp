// Copyright 2025 Weitang Ye
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
 * @FilePath: /Artea/include/artea/cpu/refiner/updaters/truncate_updater.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Truncate updater: trims each vertex's neighbor array by size and distance.
 */

#pragma once

#include <algorithm>
#include <cstddef>

namespace artea {
namespace cpu {

template <typename RefinerTraitsT>
class TruncateUpdater :
    public RefinerTraitsT::template neighbor_updater_t<TruncateUpdater<RefinerTraitsT>> {

    using vertex_id_t      = typename RefinerTraitsT::vertex_id_t;
    using vertex_num_t     = typename RefinerTraitsT::vertex_num_t;
    using distance_t       = typename RefinerTraitsT::distance_t;
    using nbr_arr_t        = typename RefinerTraitsT::nbr_arr_t;
    using log_table_t      = typename RefinerTraitsT::log_table_t;
    using dist_func_t      = typename RefinerTraitsT::dist_func_t;
    using vector_array_t   = typename RefinerTraitsT::vector_array_t;
    using refining_graph_t = typename RefinerTraitsT::dynamic::refining_graph_t;
    using base_class_t     = typename RefinerTraitsT::template neighbor_updater_t<TruncateUpdater<RefinerTraitsT>>;

public:
    static constexpr const char* updater_name = "truncate_updater";

    /** @brief Truncate to the current layer's max_nbr_size and an optional distance.
     *  @param truncate_distance Maximum retained distance (inclusive); max_distance
     *         disables distance truncation. */
    TruncateUpdater(
        const dist_func_t&        dist_func,
        const vector_array_t&     vecs_data,
        log_table_t&              log_table,
        const refining_graph_t&   refining_graph,
        distance_t                truncate_distance = RefinerTraitsT::max_distance
    ) : base_class_t(dist_func, vecs_data, log_table, refining_graph),
        _truncate_distance(truncate_distance) {}

    __attribute__((always_inline))
    auto update_impl(
        const vertex_id_t /* layer_vid */,
        nbr_arr_t& origin_nbrs
    ) -> void {
        const vertex_num_t max_sz = this->_refining_graph.layer_config().max_nbr_size();
        if (origin_nbrs.size() > max_sz) {
            origin_nbrs.resize(max_sz);
        }
        if (_truncate_distance != RefinerTraitsT::max_distance) {
            // Neighbor arrays are sorted by ascending distance.
            const auto end = std::upper_bound(
                origin_nbrs.begin(), origin_nbrs.end(), _truncate_distance,
                [](const distance_t distance, const auto& nbr) {
                    return distance < nbr.get_distance();
                });
            origin_nbrs.resize(static_cast<size_t>(end - origin_nbrs.begin()));
        }
    }

private:
    distance_t _truncate_distance;

};  // class TruncateUpdater

}   // namespace cpu
}   // namespace artea
