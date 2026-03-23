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
 * @FilePath: /Artea/include/artea/cpu/edge_generator/triangle_updater.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Triangle-based neighbor updater for edge generation.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <stdexcept>
#include <artea/cpu/utils/nbr_arr_checker.hpp>

namespace artea {
namespace cpu {

/* ------ Pruning Condition Enumeration ------ */
enum class PruningConditionT {
    scaled_ineq,
    scaled_shifted_ineq,
    shifted_ineq,
    origin_rng_ineq
};

template <typename EdgeGeneratorTraitsT>
class TriangleUpdater :
    public EdgeGeneratorTraitsT::template neighbor_updater_t<TriangleUpdater<EdgeGeneratorTraitsT>> {

    using vertex_id_t = typename EdgeGeneratorTraitsT::vertex_id_t;
    using vertex_num_t = typename EdgeGeneratorTraitsT::vertex_num_t;
    using vec_ele_t = typename EdgeGeneratorTraitsT::vec_ele_t;
    using distance_t = typename EdgeGeneratorTraitsT::distance_t;
    using ratio_t = typename EdgeGeneratorTraitsT::ratio_t;
    using vector_array_t = typename EdgeGeneratorTraitsT::vector_array_t;
    using nbr_t = typename EdgeGeneratorTraitsT::nbr_t;
    using nbr_arr_t = typename EdgeGeneratorTraitsT::nbr_arr_t;
    using log_table_t = typename EdgeGeneratorTraitsT::log_table_t;
    using dist_func_t = typename EdgeGeneratorTraitsT::dist_func_t;
    using base_class_t = typename EdgeGeneratorTraitsT::template neighbor_updater_t<TriangleUpdater<EdgeGeneratorTraitsT>>;

    static constexpr vertex_id_t invalid_vertex_id = EdgeGeneratorTraitsT::invalid_vertex_id;
    static constexpr distance_t nan_distance = EdgeGeneratorTraitsT::nan_distance;
    static constexpr distance_t max_distance = EdgeGeneratorTraitsT::max_distance;
    static constexpr bool accepted = true;
    static constexpr bool rejected = false;

public:
    static constexpr const char* updater_name = "triangle_updater";

    TriangleUpdater(
        const dist_func_t& dist_func,
        const vector_array_t& vecs_data,
        log_table_t& log_table,
        const vertex_num_t max_nbr_size,
        const ratio_t scale_coeffs,
        const ratio_t shifted_coeffs = 0.0
    ) : base_class_t(dist_func, vecs_data, log_table),
        _inv_scale_coeffs(static_cast<ratio_t>(1.0) / scale_coeffs),
        _shifted_coeffs(shifted_coeffs),
        _max_nbr_size(max_nbr_size) {}

    __attribute__((always_inline))
    auto get_max_nbr_size() const -> vertex_num_t {
        return _max_nbr_size;
    }

    __attribute__((always_inline))
    auto set_max_nbr_size(const vertex_num_t max_nbr_size) -> void {
        _max_nbr_size = max_nbr_size;
    }

    /**
     * @brief Apply triangle inequality pruning to the neighbor array of a pivot vertex.
     *
     * This operator implements the RNG (Relative Neighborhood Graph) pruning algorithm based on
     * the triangle inequality. For each neighbor candidate, it checks whether there exists a
     * "shortcut" through already-retained neighbors. If such a shortcut exists (RNG conflict),
     * the candidate is rejected and a reverse edge is logged instead.
     *
     * @tparam ConditionType The pruning condition type (default: scaled_ineq).
     *         - scaled_ineq: threshold = d_ori / scale_coeffs
     *         - scaled_shifted_ineq: threshold = d_ori / scale_coeffs - shifted_coeffs
     *         - shifted_ineq: threshold = d_ori - shifted_coeffs
     *         - origin_rng_ineq: threshold = d_ori
     *
     * @param pivot_vid The vertex ID whose neighbor array is being pruned.
     * @param origin_nbrs The neighbor array to be pruned. After pruning, this array will contain
     *                    only the retained neighbors, sorted by distance and marked as old.
     *
     * @warning origin_nbrs MUST NOT be empty before calling this operator. An empty neighbor array
     *          will cause undefined behavior in release builds and trigger an error in debug builds.
     *          The first neighbor is always retained as it represents the closest neighbor and
     *          cannot conflict with any other neighbor.
     *
     * Algorithm:
     * 1. The first (closest) neighbor is always retained without checking.
     * 2. For each subsequent neighbor, check if it conflicts with any already-retained neighbor
     *    based on the triangle inequality threshold.
     * 3. If no conflict is found, retain the neighbor; otherwise, log a reverse edge and reject it.
     * 4. Stop early if max_nbr_size is reached.
     * 5. Mark all retained neighbors as old and replace origin_nbrs with the pruned array.
     */
    template <PruningConditionT ConditionType = PruningConditionT::scaled_ineq>
    auto update_impl(
        const vertex_id_t pivot_vid,
        nbr_arr_t& origin_nbrs
    ) -> void {
        #ifndef NDEBUG
        if (origin_nbrs.empty()) {
            ARTEA_ERROR("[TriangleUpdater]: origin_nbrs cannot be empty");
        }
        #endif

        nbr_arr_t retained_nbrs;
        retained_nbrs.reserve(origin_nbrs.capacity());

        // The first neighbor is always the closest to pivot_vid and cannot conflict with any existing neighbor
        retained_nbrs.push_back(origin_nbrs[0]);

        for (vertex_num_t i = 1; i < origin_nbrs.size(); ++i) {
            const nbr_t& ori_nbr = origin_nbrs[i];
            auto [passed, conflict_vid, conflict_dist] = _internal_check<ConditionType>(ori_nbr, retained_nbrs);

            if (passed) {
                retained_nbrs.push_back(ori_nbr);
                // Do not accept because we've reached the maximum neighbor size
                if (retained_nbrs.size() >= _max_nbr_size) {
                    // break;
                    continue;
                }
            }
            else {
                vertex_id_t sacrificed_vid = ori_nbr.get_id();

                /** @brief add append operation
                  * append edge [conflict_vid -- sacrificed_vid]
                  */
                this->_log_table.write_log(
                    /* executor_vid = */conflict_vid,
                    /* nbr_id = */sacrificed_vid,
                    /* new_edge_dist = */conflict_dist
                );
            }
        }

        // Mark all retained neighbors as old before swapping
        for (vertex_num_t i = 0; i < retained_nbrs.size(); ++i) {
            retained_nbrs[i].mark_as_old();
        }

        std::swap(origin_nbrs, retained_nbrs);
    }

private:

    /** @brief Inverse of scale coefficient for RNG Triangle Inequality, used to accelerate division. */
    const ratio_t _inv_scale_coeffs;

    /** @brief Shifted coefficient for RNG Triangle Inequality. */
    const ratio_t _shifted_coeffs;

    /** @brief Maximum number of neighbors (for overflow control). */
    vertex_num_t _max_nbr_size;

    /**
     * @brief Compute the pruning threshold based on the pruning condition type.
     */
    template <PruningConditionT ConditionType>
    __attribute__((always_inline))
    constexpr auto _compute_threshold(const distance_t ori_dist) const -> distance_t {
        if constexpr (ConditionType == PruningConditionT::scaled_ineq) {
            return ori_dist * _inv_scale_coeffs;
        } else if constexpr (ConditionType == PruningConditionT::scaled_shifted_ineq) {
            return ori_dist * _inv_scale_coeffs - _shifted_coeffs;
        } else if constexpr (ConditionType == PruningConditionT::shifted_ineq) {
            return ori_dist - _shifted_coeffs;
        } else if constexpr (ConditionType == PruningConditionT::origin_rng_ineq) {
            return ori_dist;
        }
    }

    template <PruningConditionT ConditionType>
    auto _internal_check(
        const nbr_t& ori_nbr,
        const nbr_arr_t& retained_nbrs
    ) -> std::tuple<bool, vertex_id_t, distance_t> {
        const vec_ele_t* ori_vec = this->_vecs_data.get(ori_nbr.get_id());
        const distance_t threshold = _compute_threshold<ConditionType>(ori_nbr.get_distance());

        // Check conflict with all retained neighbors
        for (vertex_num_t i = 0; i < retained_nbrs.size(); ++i) {
            // Skip distance calculation for old-old pairs
            if (ori_nbr.is_old() && retained_nbrs[i].is_old()) {
                continue;
            }

            const nbr_t& retained_nbr = retained_nbrs[i];
            const vec_ele_t* retained_vec = this->_vecs_data.get(retained_nbr.get_id());
            distance_t dist_to_retained = this->_dist_func(ori_vec, retained_vec);

            if (dist_to_retained < threshold) {
                // RNG conflict detected, rejected
                return std::make_tuple(rejected, retained_nbr.get_id(), dist_to_retained);
            }
        }

        // NO RNG conflict, accepted
        return std::make_tuple(accepted, invalid_vertex_id, nan_distance);
    }

};  // class TriangleUpdater

}   // namespace cpu
}   // namespace artea
