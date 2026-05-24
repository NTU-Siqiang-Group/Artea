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
 * @FilePath: /Artea/include/artea/cpu/refiner/updaters/pruning_updater.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Pruning-based neighbor updater (no log writes). This is
 *               the only post-refining component that still consumes
 *               RNG scale/shift coefficients.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <stdexcept>
#include <artea/cpu/utils/nbr_arr_checker.hpp>

namespace artea {
namespace cpu {

/* ------ Pruning Condition Enumeration ------ *
 *
 * Only PruningUpdater consumes this enum now: the post-refining routing
 * loop uses it to pick between plain RNG and the scaled/shifted RNG
 * variants. TriangleUpdater / HierarchicalPruningUpdater hard-code the
 * plain RNG rule and ignore scale/shift entirely.
 */
enum class PruningConditionT {
    scaled_ineq,
    scaled_shifted_ineq,
    shifted_ineq,
    origin_rng_ineq
};

template <typename RefinerTraitsT, typename DistFuncT>
class PruningUpdater :
    public RefinerTraitsT::template neighbor_updater_t<DistFuncT, PruningUpdater<RefinerTraitsT, DistFuncT>> {

    using vertex_id_t = typename RefinerTraitsT::vertex_id_t;
    using vertex_num_t = typename RefinerTraitsT::vertex_num_t;
    using vec_ele_t = typename RefinerTraitsT::vec_ele_t;
    using distance_t = typename RefinerTraitsT::distance_t;
    using ratio_t = typename RefinerTraitsT::ratio_t;
    using vector_array_t = typename RefinerTraitsT::vector_array_t;
    using nbr_t = typename RefinerTraitsT::nbr_t;
    using nbr_arr_t = typename RefinerTraitsT::nbr_arr_t;
    using log_table_t = typename RefinerTraitsT::log_table_t;
    using dist_func_t = DistFuncT;
    using pruning_condition_t = typename RefinerTraitsT::pruning_condition_t;
    using refining_graph_t = typename RefinerTraitsT::dynamic::refining_graph_t;
    using base_class_t = typename RefinerTraitsT::template neighbor_updater_t<DistFuncT, PruningUpdater<RefinerTraitsT, DistFuncT>>;
    static constexpr vertex_id_t invalid_vertex_id = RefinerTraitsT::invalid_vertex_id;
    static constexpr distance_t nan_distance = RefinerTraitsT::nan_distance;
    static constexpr distance_t max_distance = RefinerTraitsT::max_distance;
    static constexpr bool accepted = true;
    static constexpr bool rejected = false;

public:
    static constexpr const char* updater_name = "pruning_updater";

    PruningUpdater(
        const dist_func_t&        dist_func,
        const vector_array_t&     vecs_data,
        log_table_t&              log_table,
        const refining_graph_t&   refining_graph,
        const ratio_t             scale_coeffs,
        const ratio_t             shifted_coeffs = 0.0
    ) : base_class_t(dist_func, vecs_data, log_table, refining_graph),
        _inv_scale_coeffs(static_cast<ratio_t>(1.0) / scale_coeffs),
        _shifted_coeffs(shifted_coeffs) {}

    __attribute__((always_inline))
    auto get_max_nbr_size() const -> vertex_num_t {
        return this->_refining_graph.layer_config().max_nbr_size();
    }

    /**
     * @brief Apply triangle inequality pruning without writing any logs.
     *
     * Same RNG pruning logic as TriangleUpdater, but rejected candidates are
     * simply discarded — no reverse edges are logged. This is useful when the
     * graph already has good edge quality and only in-place pruning is needed.
     */
    template <pruning_condition_t ConditionType = pruning_condition_t::scaled_ineq>
    auto update_impl(
        const vertex_id_t /*layer_vid*/,
        nbr_arr_t& origin_nbrs
    ) -> void {
        #ifndef NDEBUG
        if (origin_nbrs.empty()) {
            ARTEA_ERROR("[PruningUpdater]: origin_nbrs cannot be empty");
        }
        #endif

        nbr_arr_t retained_nbrs;
        retained_nbrs.reserve(origin_nbrs.capacity());
        const vertex_num_t max_sz = this->_refining_graph.layer_config().max_nbr_size();

        // The first neighbor is always the closest and cannot conflict
        retained_nbrs.push_back(origin_nbrs[0]);

        for (vertex_num_t i = 1; i < origin_nbrs.size(); ++i) {
            const nbr_t& ori_nbr = origin_nbrs[i];
            bool passed = _internal_check<ConditionType>(ori_nbr, retained_nbrs);

            if (passed) {
                retained_nbrs.push_back(ori_nbr);
                if (retained_nbrs.size() >= max_sz) {
                    continue;
                }
            }
        }

        // Mark all retained neighbors as old before swapping
        for (vertex_num_t i = 0; i < retained_nbrs.size(); ++i) {
            retained_nbrs[i].mark_as_old();
        }

        std::swap(origin_nbrs, retained_nbrs);
    }

private:

    /** @brief Inverse of scale coefficient for RNG Triangle Inequality. */
    const ratio_t _inv_scale_coeffs;

    /** @brief Shifted coefficient for RNG Triangle Inequality.
     *         Subtracted as a bare term from the candidate distance in
     *         every threshold variant that consumes the shift. */
    const ratio_t _shifted_coeffs;

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
    ) -> bool {
        const vec_ele_t* ori_vec = this->_vecs_data.get(ori_nbr.get_vid());
        const distance_t threshold = _compute_threshold<ConditionType>(ori_nbr.get_distance());

        for (vertex_num_t i = 0; i < retained_nbrs.size(); ++i) {
            if (ori_nbr.is_old() && retained_nbrs[i].is_old()) {
                continue;
            }

            const nbr_t& retained_nbr = retained_nbrs[i];
            const vec_ele_t* retained_vec = this->_vecs_data.get(retained_nbr.get_vid());
            distance_t dist_to_retained = this->_dist_func(ori_vec, retained_vec);

            if (dist_to_retained < threshold) {
                return rejected;
            }
        }

        return accepted;
    }

};  // class PruningUpdater

}   // namespace cpu
}   // namespace artea
