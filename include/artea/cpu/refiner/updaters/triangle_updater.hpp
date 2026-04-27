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
 * @FilePath: /Artea/include/artea/cpu/refiner/updaters/triangle_updater.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Triangle-based neighbor updater for edge generation.
 *               Uses two decoupled thresholds derived from the same
 *               distance: @c recommend_threshold = @c ori_dist
 *               (plain RNG) controls reverse-edge log emission, and
 *               @c prune_threshold = @c ori_dist * inv_scale - shift
 *               (scaled_shifted, same as @c PruningUpdater) controls
 *               whether @c ori_nbr is dropped from the pivot's
 *               neighbor list. Defaults (@c scale_coeffs=1,
 *               @c shifted_coeffs=0) collapse both thresholds to
 *               plain @c ori_dist.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
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
    using ratio_t = typename RefinerTraitsT::ratio_t;
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

public:
    static constexpr const char* updater_name = "triangle_updater";

    TriangleUpdater(
        const dist_func_t&        dist_func,
        const vector_array_t&     vecs_data,
        log_table_t&              log_table,
        const refining_graph_t&   refining_graph,
        const ratio_t             scale_coeffs,
        const ratio_t             shifted_coeffs
    ) : base_class_t(dist_func, vecs_data, log_table, refining_graph),
        _inv_scale_coeffs(static_cast<ratio_t>(1.0) / scale_coeffs),
        _shifted_coeffs(shifted_coeffs) {}

    __attribute__((always_inline))
    auto get_max_nbr_size() const -> vertex_num_t {
        return this->_refining_graph.layer_config().max_nbr_size();
    }

    /**
     * @brief Apply triangle-inequality pruning with decoupled
     *        recommendation logging.
     *
     * Two independent thresholds are derived from the same candidate
     * distance:
     *   - @c recommend_threshold = @c ori_dist (plain RNG). Any already-
     *     retained neighbor closer than this receives a reverse-edge
     *     log entry naming @c ori_nbr as a candidate.
     *   - @c prune_threshold    = @c ori_dist * inv_scale - shift
     *     (scaled_shifted). A retained neighbor closer than this
     *     rejects @c ori_nbr from the pivot's list.
     * Under defaults (@c scale=1, @c shift=0) both thresholds collapse
     * to @c ori_dist, recovering the plain-RNG log-on-reject behavior.
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
     * 2. For each subsequent @c ori_nbr, iterate retained neighbors;
     *    per retained @c r compute one distance and check both
     *    thresholds: emit a recommendation log on soft conflict,
     *    early-return @c rejected on a pruning conflict.
     * 3. Continue iterating even after @c max_nbr_size is reached so
     *    over-cap candidates can still contribute recommendations.
     * 4. Mark retained as old and swap into origin_nbrs.
     */
    auto update_impl(
        const vertex_id_t /*layer_vid*/,
        nbr_arr_t& origin_nbrs
    ) -> void {
        // Nothing to prune — can happen at the top layer of a sparse
        // hierarchy when the bucket holds a single vertex with no peers.
        if (origin_nbrs.empty()) return;

        nbr_arr_t retained_nbrs;
        retained_nbrs.reserve(origin_nbrs.capacity());
        const vertex_num_t max_sz = this->_refining_graph.layer_config().max_nbr_size();

        // The first neighbor is always the closest to pivot_vid and cannot conflict with any existing neighbor
        retained_nbrs.push_back(origin_nbrs[0]);

        for (vertex_num_t i = 1; i < origin_nbrs.size(); ++i) {
            const nbr_t& ori_nbr = origin_nbrs[i];
            const auto [recommend_to, will_prune] = _internal_check(ori_nbr, retained_nbrs);

            if (!recommend_to.is_invalid()) {
                this->_log_table.write_log(
                    this->_refining_graph.local_id_of(recommend_to.get_vid()),
                    ori_nbr.get_vid(),
                    recommend_to.get_distance()
                );
            }

            if (!will_prune) {
                retained_nbrs.push_back(ori_nbr);
                // Do not accept because we've reached the maximum neighbor size
                if (retained_nbrs.size() >= max_sz) {
                    // break;
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
     *         Subtracted as a bare term from the scaled distance in the
     *         prune-threshold formula. */
    const ratio_t _shifted_coeffs;

    /**
     * @brief Pure conflict scan. No side effects: the caller owns all
     *        log-table writes.
     *
     *        Two decoupled thresholds derived from
     *        @c checking_nbr.get_distance() :
     *          - @c recommend_threshold = @c checking_dist
     *          - @c prune_threshold     = @c checking_dist * inv_scale - shift
     *
     *        Iterates @p retained_nbrs once. At most one recommendation
     *        target is reported per call — the retained neighbor with
     *        the smallest @c dist_to_retained among those crossing
     *        @c recommend_threshold (up to the point the scan stops).
     *        Any retained with @c dist_to_retained < prune_threshold
     *        flips the prune flag and stops the scan.
     *
     *        Old-old pairs short-circuit without a distance call — they
     *        were fully evaluated in a prior iteration. This is also
     *        what prevents re-emitting the same recommendation across
     *        iterations, since @c update_impl marks every retained
     *        neighbor old before returning.
     *
     * @param checking_nbr  Candidate being checked.
     * @param retained_nbrs Already-retained neighbors to scan against.
     * @return @c {recommend_to, will_prune}
     *           - @c recommend_to : target of the reverse-edge
     *             recommendation, carrying (vid, dist_to_retained).
     *             @c is_invalid() means no recommendation was found.
     *           - @c will_prune   : @c true iff some retained had
     *             @c dist_to_retained < prune_threshold (drop @c checking_nbr).
     */
    __attribute__((always_inline))
    auto _internal_check(
        const nbr_t&     checking_nbr,
        const nbr_arr_t& retained_nbrs
    ) -> std::pair<nbr_t, bool> {
        const vec_ele_t* checking_vec = this->_vecs_data.get(checking_nbr.get_vid());
        const distance_t recommend_threshold = checking_nbr.get_distance();
        const distance_t prune_threshold     = checking_nbr.get_distance() * _inv_scale_coeffs - _shifted_coeffs;

        // recommend_to starts invalid (distance = max_distance), so the
        // "closer-than-current" test below picks up the first soft
        // conflict automatically and then only updates on strictly
        // closer retained neighbors.
        nbr_t recommend_to = nbr_t::make_invalid_nbr();
        bool  will_prune   = false;

        for (vertex_num_t i = 0; i < retained_nbrs.size(); ++i) {
            if (checking_nbr.is_old() && retained_nbrs[i].is_old()) {
                continue;
            }

            const nbr_t& retained_nbr = retained_nbrs[i];
            const vec_ele_t* retained_vec = this->_vecs_data.get(retained_nbr.get_vid());
            const distance_t dist_to_retained = this->_dist_func(checking_vec, retained_vec);

            if (dist_to_retained < recommend_threshold && dist_to_retained < recommend_to.get_distance()) {
                recommend_to = nbr_t(retained_nbr.get_vid(), dist_to_retained);
            }
            if (dist_to_retained < prune_threshold) {
                will_prune = true;
                break;
            }
        }

        return {recommend_to, will_prune};
    }

};  // class TriangleUpdater

}   // namespace cpu
}   // namespace artea
