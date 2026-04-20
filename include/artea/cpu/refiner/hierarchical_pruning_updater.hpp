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
 * @Description: Pure-RNG neighbor pruning over nbr_t, decoupled from any
 *               RefiningGraph / layer_config. Used by
 *               stacked_rgraph::IndexFactory for both forward-edge
 *               pruning (new vertex's candidate list) and reverse-edge
 *               pruning (existing neighbors' lists under overflow).
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include <artea/common/logger.hpp>

namespace artea {
namespace cpu {

/**
 * @brief In-place pure-RNG neighbor pruning over @c nbr_t.
 *
 * Accepts a distance-ascending sorted list of candidates and retains at
 * most @c max_nbr_size entries that pass the plain triangle-inequality
 * check (threshold = ori_dist, no scale/shift) against already-retained
 * entries.
 *
 * Unlike the RefiningGraph-centric @c PruningUpdater, this class is
 * stateless except for the references to @c vecs_data + @c dist_func
 * needed to compute neighbor-to-neighbor distances. @c max_nbr_size is a
 * per-call parameter because the hierarchical graph uses different
 * capacities per level (level 0 has 2 * max_nbr_size, upper levels have
 * max_nbr_size).
 *
 * @tparam RefinerTraitsT The refiner traits type.
 */
template <typename RefinerTraitsT>
class HierarchicalPruningUpdater {

    using vertex_num_t    = typename RefinerTraitsT::vertex_num_t;
    using vec_ele_t       = typename RefinerTraitsT::vec_ele_t;
    using distance_t      = typename RefinerTraitsT::distance_t;
    using ratio_t         = typename RefinerTraitsT::ratio_t;
    using vector_array_t  = typename RefinerTraitsT::vector_array_t;
    using dist_func_t     = typename RefinerTraitsT::dist_func_t;
    using nbr_t           = typename RefinerTraitsT::nbr_t;

public:
    static constexpr const char* updater_name = "hierarchical_pruning_updater";

    /**
     * @brief Construct an updater bound to a vector array + distance
     *        functor. The scale coefficient applied to the RNG triangle
     *        threshold is passed per @c update_impl call, so a single
     *        updater instance can serve both plain (L0) and scaled
     *        (L1+) callers without re-binding state.
     *
     * @param dist_func  Distance functor.
     * @param vecs_data  Base vector storage (coordinate lookup).
     */
    HierarchicalPruningUpdater(
        const dist_func_t&    dist_func,
        const vector_array_t& vecs_data
    ) :
        _dist_func(dist_func),
        _vecs_data(vecs_data) {}

    /**
     * @brief Prune @p origin_nbrs in place, keeping at most
     *        @p max_nbr_size entries.
     *
     * @p origin_nbrs must be sorted ascending by distance on entry. The
     * first entry is always retained (it is the closest); each
     * subsequent entry is retained iff every already-retained entry is
     * strictly farther from it than @c (ori_dist / scale_coeffs). With
     * @p scale_coeffs == 1 this collapses to the plain RNG rule; values
     * > 1 relax the triangle check and let more neighbors through.
     * Retained entries preserve their ascending order and are marked as
     * "old".
     *
     * @param origin_nbrs   Candidate list; pruned in place.
     * @param max_nbr_size  Maximum number of retained neighbors.
     * @param scale_coeffs  RNG scale coefficient (default 1.0 = plain
     *                      RNG). Must be > 0.
     */
    auto update_impl(
        std::vector<nbr_t>& origin_nbrs,
        const vertex_num_t  max_nbr_size,
        const ratio_t       scale_coeffs = ratio_t(1)
    ) const -> void {
        if (origin_nbrs.empty()) return;

        std::vector<nbr_t> retained_nbrs;
        retained_nbrs.reserve(
            std::min<std::size_t>(origin_nbrs.size(), max_nbr_size));

        // The closest candidate is always retained.
        retained_nbrs.push_back(origin_nbrs[0]);

        const ratio_t inv_scale = ratio_t(1) / scale_coeffs;

        for (vertex_num_t i = 1;
             i < origin_nbrs.size() && retained_nbrs.size() < max_nbr_size;
             ++i)
        {
            const nbr_t& ori_nbr = origin_nbrs[i];
            const distance_t threshold =
                static_cast<distance_t>(ori_nbr.get_distance() * inv_scale);
            const vec_ele_t* ori_vec   = _vecs_data.get(ori_nbr.get_vid());

            bool accepted = true;
            for (const nbr_t& retained_nbr : retained_nbrs) {
                // Old/old pairs can skip the triangle check: neither
                // changed since the last refinement round, so their
                // prior decision still holds.
                if (ori_nbr.is_old() && retained_nbr.is_old()) continue;

                const vec_ele_t* retained_vec = _vecs_data.get(retained_nbr.get_vid());
                const distance_t dist_to_retained = _dist_func(ori_vec, retained_vec);
                if (dist_to_retained < threshold) { accepted = false; break; }
            }
            if (accepted) retained_nbrs.push_back(ori_nbr);
        }

        // Mark retained as "old" — matches PruningUpdater convention.
        for (nbr_t& nbr : retained_nbrs) {
            nbr.mark_as_old();
        }

        std::swap(origin_nbrs, retained_nbrs);
    }

private:
    const dist_func_t&    _dist_func;
    const vector_array_t& _vecs_data;

};  // class HierarchicalPruningUpdater

}   // namespace cpu
}   // namespace artea
