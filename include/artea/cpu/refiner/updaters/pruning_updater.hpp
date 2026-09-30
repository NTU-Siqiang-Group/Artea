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
#include <cmath>
#include <vector>
#include <stdexcept>
#include <artea/cpu/utils/nbr_arr_checker.hpp>

namespace artea {
namespace cpu {

/* ------ Pruning Condition Enumeration ------ *
 *
 * Shared by the graph-bound PruningUpdater and standalone candidate pruning.
 */
enum class PruningConditionT {
    scaled_ineq,
    scaled_shifted_ineq,
    shifted_ineq,
    origin_rng_ineq
};

/**
 * @brief Prune a distance-sorted candidate list without graph storage or a degree cap.
 *
 * The distance functor and vectors must outlive this updater. Each invocation
 * owns its output, so distinct lists can be processed concurrently with a
 * thread-safe distance functor. Retained records are marked OLD only at the end.
 * SkipOldPairs preserves the incremental refiner optimization; exact construction
 * leaves it false and checks all pairs regardless of their labels.
 */
template <typename TraitsT, PruningConditionT ConditionType, bool SkipOldPairs = false>
class CandidatePruningUpdater {
    using nbr_arr_t = typename TraitsT::nbr_arr_t;
    using distance_t = typename TraitsT::distance_t;
    using ratio_t = typename TraitsT::ratio_t;
    using vector_array_t = typename TraitsT::vector_array_t;
    using dist_func_t = typename TraitsT::dist_func_t;

public:
    CandidatePruningUpdater(const dist_func_t& dist_func, const vector_array_t& vectors,
                            ratio_t scale_coeffs, ratio_t shifted_coeffs = 0)
        : _dist_func(dist_func), _vectors(vectors),
          _inv_scale_coeffs(ratio_t(1) / scale_coeffs), _shifted_coeffs(shifted_coeffs) {}

    auto operator()(nbr_arr_t& candidates) const -> void {
        if (candidates.empty()) return;
        nbr_arr_t retained;
        retained.reserve(candidates.capacity());
        for (const auto& candidate : candidates) {
            const auto* vector = _vectors.get(candidate.get_vid());
            const distance_t threshold = compute_threshold(candidate.get_distance());
            bool keep = true;
            for (const auto& neighbor : retained) {
                if constexpr (SkipOldPairs) {
                    if (candidate.is_old() && neighbor.is_old()) continue;
                }
                const distance_t distance = _dist_func(vector, _vectors.get(neighbor.get_vid()));
                if constexpr (!SkipOldPairs) {
                    if (!std::isfinite(distance) || distance < distance_t(0)) {
                        throw std::domain_error("Exact pruning requires finite nonnegative distances");
                    }
                }
                if (distance < threshold) {
                    keep = false;
                    break;
                }
            }
            if (keep) retained.push_back(candidate);
        }
        for (auto& neighbor : retained) neighbor.mark_as_old();
        candidates.swap(retained);
    }

private:
    auto compute_threshold(distance_t distance) const -> distance_t {
        if constexpr (ConditionType == PruningConditionT::scaled_ineq) {
            return distance * _inv_scale_coeffs;
        } else if constexpr (ConditionType == PruningConditionT::scaled_shifted_ineq) {
            return distance * _inv_scale_coeffs - _shifted_coeffs;
        } else if constexpr (ConditionType == PruningConditionT::shifted_ineq) {
            return distance - _shifted_coeffs;
        } else {
            static_assert(ConditionType == PruningConditionT::origin_rng_ineq);
            return distance;
        }
    }

    const dist_func_t& _dist_func;
    const vector_array_t& _vectors;
    const ratio_t _inv_scale_coeffs;
    const ratio_t _shifted_coeffs;
};

template <typename RefinerTraitsT>
class PruningUpdater :
    public RefinerTraitsT::template neighbor_updater_t<PruningUpdater<RefinerTraitsT>> {

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
    using pruning_condition_t = typename RefinerTraitsT::pruning_condition_t;
    using refining_graph_t = typename RefinerTraitsT::dynamic::refining_graph_t;
    using base_class_t = typename RefinerTraitsT::template neighbor_updater_t<PruningUpdater<RefinerTraitsT>>;
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
        _scale_coeffs(scale_coeffs),
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
        CandidatePruningUpdater<RefinerTraitsT, ConditionType, true> prune(
            this->_dist_func, this->_vecs_data, _scale_coeffs, _shifted_coeffs);
        prune(origin_nbrs);
    }

private:

    /** @brief Scale coefficient for RNG Triangle Inequality. */
    const ratio_t _scale_coeffs;

    /** @brief Shifted coefficient for RNG Triangle Inequality.
     *         Subtracted as a bare term from the candidate distance in
     *         every threshold variant that consumes the shift. */
    const ratio_t _shifted_coeffs;

};  // class PruningUpdater

}   // namespace cpu
}   // namespace artea
