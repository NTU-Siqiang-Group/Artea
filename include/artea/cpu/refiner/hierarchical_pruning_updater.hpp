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
 * @FilePath: /Artea/include/artea/cpu/refiner/hierarchical_pruning_updater.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: RNG triangle-inequality pruning over nbr_t, decoupled
 *               from any BottomGraph / layer_config. Used by
 *               stacked_rgraph::IndexFactory for both forward-edge
 *               pruning (new vertex's candidate list) and reverse-edge
 *               pruning (existing neighbors' lists under overflow).
 *
 *               TODO: the RNG logic is duplicated from PruningUpdater.
 *                     Once both hierarchical and bottom updaters have
 *                     stabilized, unify them on a common kernel.
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include <artea/common/logger.hpp>
#include <artea/cpu/refiner/triangle_updater.hpp>   // for PruningConditionT

namespace artea {
namespace cpu {

/**
 * @brief In-place RNG-style neighbor pruning over @c nbr_t.
 *
 * Accepts a distance-ascending sorted list of candidates and retains at
 * most @c max_nbr_size entries that pass the scaled triangle-inequality
 * check against already-retained entries.
 *
 * Unlike the BottomGraph-centric @c PruningUpdater, this class is stateless
 * except for the pruning coefficients and the references to
 * @c vecs_data + @c dist_func needed to compute neighbor-to-neighbor
 * distances. @c max_nbr_size is a per-call parameter because the
 * hierarchical graph uses different capacities per level
 * (level 0 has 2 * max_nbr_size, upper levels have max_nbr_size).
 *
 * @tparam RefinerTraitsT The refiner traits type.
 */
template <typename RefinerTraitsT>
class HierarchicalPruningUpdater {

    using vertex_id_t     = typename RefinerTraitsT::vertex_id_t;
    using vertex_num_t    = typename RefinerTraitsT::vertex_num_t;
    using vec_ele_t       = typename RefinerTraitsT::vec_ele_t;
    using distance_t      = typename RefinerTraitsT::distance_t;
    using ratio_t         = typename RefinerTraitsT::ratio_t;
    using vector_array_t  = typename RefinerTraitsT::vector_array_t;
    using dist_func_t     = typename RefinerTraitsT::dist_func_t;
    using nbr_t           = typename RefinerTraitsT::nbr_t;

    static constexpr bool accepted = true;
    static constexpr bool rejected = false;

public:
    static constexpr const char* updater_name = "hierarchical_pruning_updater";

    /**
     * @brief Construct an updater bound to a vector array + distance
     *        functor + RNG coefficients.
     *
     * @param dist_func        Distance functor.
     * @param vecs_data        Base vector storage (coordinate lookup).
     * @param scale_coeffs     RNG scale; threshold = ori_dist / scale.
     * @param shifted_coeffs   RNG shift; used only by shifted variants.
     */
    HierarchicalPruningUpdater(
        const dist_func_t&    dist_func,
        const vector_array_t& vecs_data,
        const ratio_t         scale_coeffs,
        const ratio_t         shifted_coeffs = ratio_t(0)
    ) :
        _dist_func(dist_func),
        _vecs_data(vecs_data),
        _inv_scale_coeffs(static_cast<ratio_t>(1.0) / scale_coeffs),
        _shifted_coeffs(shifted_coeffs) {}

    /**
     * @brief Prune @p origin_nbrs in place, keeping at most
     *        @p max_nbr_size entries.
     *
     * @p origin_nbrs must be sorted ascending by distance on entry. The
     * first entry is always retained (it is the closest); each
     * subsequent entry is retained iff it is strictly farther from every
     * already-retained entry than it is from the pivot (up to the
     * configured scale/shift). Retained entries preserve their ascending
     * order and are marked as "old".
     *
     * @param pivot_vid     The vid whose neighbor list is being pruned.
     *                      Used only for the neighbor-to-pivot distance
     *                      lookup (via @p origin_nbrs' recorded
     *                      distances, not via @c _dist_func).
     * @param origin_nbrs   Candidate list; pruned in place.
     * @param max_nbr_size  Maximum number of retained neighbors.
     */
    template <PruningConditionT ConditionType = PruningConditionT::scaled_ineq>
    auto update_impl(
        const vertex_id_t   pivot_vid,
        std::vector<nbr_t>& origin_nbrs,
        const vertex_num_t  max_nbr_size
    ) const -> void {
        (void)pivot_vid;   // reserved for future debug / diagnostics
        if (origin_nbrs.empty()) return;

        std::vector<nbr_t> retained_nbrs;
        retained_nbrs.reserve(
            std::min<std::size_t>(origin_nbrs.size(), max_nbr_size));

        // The closest candidate is always retained.
        retained_nbrs.push_back(origin_nbrs[0]);

        for (vertex_num_t i = 1; i < origin_nbrs.size() && retained_nbrs.size() < max_nbr_size; ++i) {
            const nbr_t& ori_nbr = origin_nbrs[i];
            if (_internal_check<ConditionType>(ori_nbr, retained_nbrs)) {
                retained_nbrs.push_back(ori_nbr);
            }
        }

        // Mark retained as "old" — matches PruningUpdater convention.
        for (nbr_t& nbr : retained_nbrs) {
            nbr.mark_as_old();
        }

        std::swap(origin_nbrs, retained_nbrs);
    }

private:
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
        const nbr_t&              ori_nbr,
        const std::vector<nbr_t>& retained_nbrs
    ) const -> bool {
        const vec_ele_t* ori_vec = _vecs_data.get(ori_nbr.get_vid());
        const distance_t threshold = _compute_threshold<ConditionType>(ori_nbr.get_distance());

        for (const nbr_t& retained_nbr : retained_nbrs) {
            // Matches PruningUpdater: if both candidate and retained are
            // marked as "old" (i.e. unchanged since last refinement
            // round), skip the triangle check to save work.
            if (ori_nbr.is_old() && retained_nbr.is_old()) continue;

            const vec_ele_t* retained_vec = _vecs_data.get(retained_nbr.get_vid());
            const distance_t dist_to_retained = _dist_func(ori_vec, retained_vec);
            if (dist_to_retained < threshold) return rejected;
        }

        return accepted;
    }

    const dist_func_t&    _dist_func;
    const vector_array_t& _vecs_data;
    const ratio_t         _inv_scale_coeffs;
    const ratio_t         _shifted_coeffs;

};  // class HierarchicalPruningUpdater

}   // namespace cpu
}   // namespace artea
