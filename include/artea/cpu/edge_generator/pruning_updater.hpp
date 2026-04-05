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
 * @FilePath: /Artea/include/artea/cpu/edge_generator/pruning_updater.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Pruning-based neighbor updater (no log writes).
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <stdexcept>
#include <artea/cpu/utils/nbr_arr_checker.hpp>

namespace artea {
namespace cpu {

template <typename EdgeGeneratorTraitsT, typename FlatGraphT>
class PruningUpdater :
    public EdgeGeneratorTraitsT::template neighbor_updater_t<FlatGraphT, PruningUpdater<EdgeGeneratorTraitsT, FlatGraphT>> {

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
    using pruning_condition_t = typename EdgeGeneratorTraitsT::pruning_condition_t;
    using base_class_t = typename EdgeGeneratorTraitsT::template neighbor_updater_t<FlatGraphT, PruningUpdater<EdgeGeneratorTraitsT, FlatGraphT>>;
    static constexpr vertex_id_t invalid_vertex_id = EdgeGeneratorTraitsT::invalid_vertex_id;
    static constexpr distance_t nan_distance = EdgeGeneratorTraitsT::nan_distance;
    static constexpr distance_t max_distance = EdgeGeneratorTraitsT::max_distance;
    static constexpr bool accepted = true;
    static constexpr bool rejected = false;

public:
    static constexpr const char* updater_name = "pruning_updater";

    PruningUpdater(
        const dist_func_t& dist_func,
        const vector_array_t& vecs_data,
        log_table_t& log_table,
        const FlatGraphT& flat_graph,
        const ratio_t scale_coeffs,
        const ratio_t shifted_coeffs = 0.0
    ) : base_class_t(dist_func, vecs_data, log_table, flat_graph),
        _inv_scale_coeffs(static_cast<ratio_t>(1.0) / scale_coeffs),
        _shifted_coeffs(shifted_coeffs) {}

    __attribute__((always_inline))
    auto get_max_nbr_size() const -> vertex_num_t {
        return this->_flat_graph.layer_config().max_nbr_size();
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
        const vertex_id_t pivot_vid,
        nbr_arr_t& origin_nbrs
    ) -> void {
        #ifndef NDEBUG
        if (origin_nbrs.empty()) {
            ARTEA_ERROR("[PruningUpdater]: origin_nbrs cannot be empty");
        }
        #endif

        nbr_arr_t retained_nbrs;
        retained_nbrs.reserve(origin_nbrs.capacity());
        const vertex_num_t max_sz = this->_flat_graph.layer_config().max_nbr_size();

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

    /** @brief Shifted coefficient for RNG Triangle Inequality. */
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
        const vec_ele_t* ori_vec = this->_vecs_data.get(ori_nbr.get_id());
        const distance_t threshold = _compute_threshold<ConditionType>(ori_nbr.get_distance());

        for (vertex_num_t i = 0; i < retained_nbrs.size(); ++i) {
            if (ori_nbr.is_old() && retained_nbrs[i].is_old()) {
                continue;
            }

            const nbr_t& retained_nbr = retained_nbrs[i];
            const vec_ele_t* retained_vec = this->_vecs_data.get(retained_nbr.get_id());
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
