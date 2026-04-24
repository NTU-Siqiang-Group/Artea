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
 * @FilePath: /Artea/include/artea/cpu/refiner/updaters/arc_updater.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Arc-radius pruning updater. Drops every neighbor whose
 *               edge length exceeds @c _arc_threshold. No log writes.
 *               @c _arc_threshold is mutable via @c set_arc_threshold, so the
 *               same instance can be re-used across layers that carry
 *               different covering radii.
 */

#pragma once

#include <cstddef>

namespace artea {
namespace cpu {

template <typename RefinerTraitsT>
class ARCUpdater :
    public RefinerTraitsT::template neighbor_updater_t<ARCUpdater<RefinerTraitsT>> {

    using vertex_id_t      = typename RefinerTraitsT::vertex_id_t;
    using vertex_num_t     = typename RefinerTraitsT::vertex_num_t;
    using distance_t       = typename RefinerTraitsT::distance_t;
    using nbr_t            = typename RefinerTraitsT::nbr_t;
    using nbr_arr_t        = typename RefinerTraitsT::nbr_arr_t;
    using log_table_t      = typename RefinerTraitsT::log_table_t;
    using dist_func_t      = typename RefinerTraitsT::dist_func_t;
    using vector_array_t   = typename RefinerTraitsT::vector_array_t;
    using refining_graph_t = typename RefinerTraitsT::dynamic::refining_graph_t;
    using base_class_t     = typename RefinerTraitsT::template neighbor_updater_t<ARCUpdater<RefinerTraitsT>>;

public:
    static constexpr const char* updater_name = "arc_updater";

    ARCUpdater(
        const dist_func_t&        dist_func,
        const vector_array_t&     vecs_data,
        log_table_t&              log_table,
        const refining_graph_t&   refining_graph,
        const distance_t          arc_threshold = distance_t(0)
    ) : base_class_t(dist_func, vecs_data, log_table, refining_graph),
        _arc_threshold(arc_threshold) {}

    __attribute__((always_inline))
    auto arc_threshold() const -> distance_t { return _arc_threshold; }

    __attribute__((always_inline))
    auto set_arc_threshold(const distance_t arc_threshold) -> void { _arc_threshold = arc_threshold; }

    /**
     * @brief Discard every neighbor whose stored edge length is strictly
     *        greater than @c _arc_threshold. Order-preserving in-place
     *        compaction — makes no assumption about @p origin_nbrs being
     *        sorted.
     */
    __attribute__((always_inline))
    auto update_impl(const vertex_id_t /*layer_vid*/, nbr_arr_t& origin_nbrs) -> void {
        vertex_num_t write = 0;
        for (vertex_num_t read = 0; read < origin_nbrs.size(); ++read) {
            if (origin_nbrs[read].get_distance() <= _arc_threshold) {
                if (write != read) origin_nbrs[write] = origin_nbrs[read];
                ++write;
            }
        }
        if (write < origin_nbrs.size()) origin_nbrs.resize(write);
    }

private:
    distance_t _arc_threshold;

};  // class ARCUpdater

}   // namespace cpu
}   // namespace artea
