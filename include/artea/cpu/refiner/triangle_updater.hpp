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
 * @FilePath: /Artea/include/artea/cpu/refiner/triangle_updater.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Pure-RNG triangle-based neighbor updater for edge
 *               generation. Threshold is always @c ori_dist — no
 *               scale/shift coefficients are consumed. Conflicts write
 *               reverse-edge entries to the log table.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <vector>
#include <stdexcept>
#include <artea/cpu/utils/nbr_arr_checker.hpp>

namespace artea {
namespace cpu {

template <typename RefinerTraitsT>
class TriangleUpdater :
    public RefinerTraitsT::template neighbor_updater_t<TriangleUpdater<RefinerTraitsT>> {

    using vertex_id_t = typename RefinerTraitsT::vertex_id_t;
    using vertex_num_t = typename RefinerTraitsT::vertex_num_t;
    using vec_ele_t = typename RefinerTraitsT::vec_ele_t;
    using distance_t = typename RefinerTraitsT::distance_t;
    using vector_array_t = typename RefinerTraitsT::vector_array_t;
    using nbr_t = typename RefinerTraitsT::nbr_t;
    using nbr_arr_t = typename RefinerTraitsT::nbr_arr_t;
    using log_table_t = typename RefinerTraitsT::log_table_t;
    using dist_func_t = typename RefinerTraitsT::dist_func_t;
    using refining_graph_t = typename RefinerTraitsT::dynamic::refining_graph_t;
    using base_class_t = typename RefinerTraitsT::template neighbor_updater_t<TriangleUpdater<RefinerTraitsT>>;
    static constexpr vertex_id_t invalid_vertex_id = RefinerTraitsT::invalid_vertex_id;
    static constexpr distance_t nan_distance = RefinerTraitsT::nan_distance;
    static constexpr distance_t max_distance = RefinerTraitsT::max_distance;
    static constexpr bool accepted = true;
    static constexpr bool rejected = false;

public:
    static constexpr const char* updater_name = "triangle_updater";

    TriangleUpdater(
        const dist_func_t&        dist_func,
        const vector_array_t&     vecs_data,
        log_table_t&              log_table,
        const refining_graph_t&   refining_graph
    ) : base_class_t(dist_func, vecs_data, log_table, refining_graph) {}

    __attribute__((always_inline))
    auto get_max_nbr_size() const -> vertex_num_t {
        return this->_refining_graph.layer_config().max_nbr_size();
    }

    /**
     * @brief Apply pure-RNG triangle-inequality pruning to the neighbor
     *        array of a pivot vertex.
     *
     * For each candidate, the threshold is simply its own distance to
     * the pivot (@c ori_dist): a candidate is rejected iff some already-
     * retained neighbor is closer to it than the pivot is. Rejected
     * candidates log a reverse-edge entry rather than being silently
     * discarded.
     *
     * @param pivot_vid   Unused; retained only for the updater interface.
     * @param origin_nbrs The neighbor array to be pruned. On return it
     *                    contains only retained neighbors, sorted by
     *                    distance and marked as old.
     *
     * @warning origin_nbrs MUST NOT be empty before calling this
     *          operator. Debug builds assert; release builds have UB.
     *
     * Algorithm:
     * 1. The first (closest) neighbor is always retained.
     * 2. For each subsequent neighbor, check if it conflicts with any
     *    already-retained neighbor under the plain RNG rule.
     * 3. Retain on success; otherwise log a reverse edge and reject.
     * 4. Stop early once max_nbr_size is reached.
     * 5. Mark retained as old and swap into origin_nbrs.
     */
    auto update_impl(
        const vertex_id_t /*layer_vid*/,
        nbr_arr_t& origin_nbrs
    ) -> void {
        #ifndef NDEBUG
        if (origin_nbrs.empty()) {
            ARTEA_ERROR("[TriangleUpdater]: origin_nbrs cannot be empty");
        }
        #endif

        nbr_arr_t retained_nbrs;
        retained_nbrs.reserve(origin_nbrs.capacity());
        const vertex_num_t max_sz = this->_refining_graph.layer_config().max_nbr_size();

        // The first neighbor is always the closest to pivot_vid and cannot conflict with any existing neighbor
        retained_nbrs.push_back(origin_nbrs[0]);

        for (vertex_num_t i = 1; i < origin_nbrs.size(); ++i) {
            const nbr_t& ori_nbr = origin_nbrs[i];
            auto [passed, conflict_vid, conflict_dist] = _internal_check(ori_nbr, retained_nbrs);

            if (passed) {
                retained_nbrs.push_back(ori_nbr);
                // Do not accept because we've reached the maximum neighbor size
                if (retained_nbrs.size() >= max_sz) {
                    // break;
                    continue;
                }
            }
            else {
                // conflict_vid is a global nbr vid; translate to local for
                // the log_table (row index into the RG).
                this->_log_table.write_log(
                    this->_refining_graph.local_id_of(conflict_vid),
                    ori_nbr.get_vid(), conflict_dist);
            }
        }

        // Mark all retained neighbors as old before swapping
        for (vertex_num_t i = 0; i < retained_nbrs.size(); ++i) {
            retained_nbrs[i].mark_as_old();
        }

        std::swap(origin_nbrs, retained_nbrs);
    }

private:

    auto _internal_check(
        const nbr_t& ori_nbr,
        const nbr_arr_t& retained_nbrs
    ) -> std::tuple<bool, vertex_id_t, distance_t> {
        const vec_ele_t* ori_vec = this->_vecs_data.get(ori_nbr.get_vid());
        const distance_t threshold = ori_nbr.get_distance();

        // Check conflict with all retained neighbors
        for (vertex_num_t i = 0; i < retained_nbrs.size(); ++i) {
            // Skip distance calculation for old-old pairs
            if (ori_nbr.is_old() && retained_nbrs[i].is_old()) {
                continue;
            }

            const nbr_t& retained_nbr = retained_nbrs[i];
            const vec_ele_t* retained_vec = this->_vecs_data.get(retained_nbr.get_vid());
            distance_t dist_to_retained = this->_dist_func(ori_vec, retained_vec);

            if (dist_to_retained < threshold) {
                // RNG conflict detected, rejected
                return std::make_tuple(rejected, retained_nbr.get_vid(), dist_to_retained);
            }
        }

        // NO RNG conflict, accepted
        return std::make_tuple(accepted, invalid_vertex_id, nan_distance);
    }

};  // class TriangleUpdater

}   // namespace cpu
}   // namespace artea
