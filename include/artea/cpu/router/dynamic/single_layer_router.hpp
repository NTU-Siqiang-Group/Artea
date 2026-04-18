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
 * @FilePath: /Artea/include/artea/cpu/router/dynamic/single_layer_router.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Single-level beam/greedy atom over dynamic::HierarchicalGraph.
 *               Apex sampling lives in CandidateSampleUtils, not here.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>

#include <artea/common/logger.hpp>
#include <artea/cpu/router/detail/beam_loop.hpp>
#include <artea/cpu/router/detail/dynamic_layer_range.hpp>

namespace artea {
namespace cpu {
namespace dynamic {

/**
 * @brief Single-level beam-search primitive for
 *        @c dynamic::HierarchicalGraph. Exposes @c beam_search and
 *        @c greedy_search only — apex sampling is provided by
 *        @c CandidateSampleUtils so callers can swap sampling policies
 *        without dragging RNG state into the router.
 *
 * @tparam RouterTraitsT The router traits type.
 */
template <typename RouterTraitsT>
class SingleLayerRouter {

    using vertex_num_t          = typename RouterTraitsT::vertex_num_t;
    using vertex_id_t           = typename RouterTraitsT::vertex_id_t;
    using layer_id_t            = typename RouterTraitsT::layer_id_t;
    using vec_ele_t             = typename RouterTraitsT::vec_ele_t;
    using distance_t            = typename RouterTraitsT::distance_t;
    using dist_func_t           = typename RouterTraitsT::dist_func_t;
    using vector_array_t        = typename RouterTraitsT::vector_array_t;
    using nbr_t                 = typename RouterTraitsT::nbr_t;
    using candidate_entry_t     = typename RouterTraitsT::candidate_entry_t;
    using std_candidate_queue_t = typename RouterTraitsT::std_candidate_queue_t;
    using visited_table_t       = typename RouterTraitsT::visited_table_t;

    static constexpr vertex_id_t invalid_vertex_id = RouterTraitsT::invalid_vertex_id;

public:
    SingleLayerRouter(
        const vector_array_t& base_vecs,
        const dist_func_t&    dist_func
    ) :
        _vecs_data(base_vecs),
        _dist_func(dist_func) {}

    /**
     * @brief Greedy best-improvement walk on one level of @p hier_graph.
     *        Uses @p visited to skip distance computation for vids that
     *        were already evaluated earlier on the same walk (a common
     *        neighbor of multiple cursor positions). The caller owns the
     *        clear/reset policy; this method only seeds @p visited with
     *        @p seed_vid before the walk.
     */
    template <typename HierarchicalGraphT>
    __attribute__((always_inline))
    auto greedy_search(
        const vec_ele_t*          query_vec,
        const HierarchicalGraphT& hier_graph,
        const layer_id_t          level_id,
        vertex_id_t               seed_vid,
        distance_t                seed_dist,
        visited_table_t&          visited
    ) const -> std::pair<vertex_id_t, distance_t> {
        visited.set(seed_vid);
        const detail::DynamicLayerRange<HierarchicalGraphT> nbrs_range(hier_graph, level_id);
        return detail::greedy_loop_body<RouterTraitsT>(
            query_vec, nbrs_range, seed_vid, seed_dist,
            visited, _dist_func, _vecs_data);
    }

    /**
     * @brief Beam search on one level of @p hier_graph, operating in-place on a
     *        pre-seeded candidate queue. Does NOT clear @p visited — the
     *        caller owns clear lifecycle so visited can be shared across
     *        hierarchical layers (and across the insert descent + select
     *        phases). Only marks the current queue seeds as visited,
     *        which is idempotent for carry-over seeds.
     */
    template <typename HierarchicalGraphT>
    __attribute__((always_inline))
    auto beam_search(
        const vec_ele_t*          query_vec,
        const HierarchicalGraphT& hier_graph,
        const layer_id_t          level_id,
        std_candidate_queue_t&    candidate_queue,
        visited_table_t&          visited
    ) const -> void {
        // Re-mark every existing candidate as unexplored. In a fresh
        // single-layer call this is a no-op, but under shared-queue
        // hierarchical descent (same queue across cur=top..1), candidates
        // popped at an upper layer would otherwise never be expanded via
        // this layer's edges because pop_best_unexplored permanently
        // removes them from the unexplored heap. That silent loss causes
        // min_dist_per_level to under-reflect the true NN and wrecks
        // subsequent absorption decisions.
        //
        // MUST run BEFORE the empty() check: empty() inspects
        // _unexplored_set, which is fully drained after the previous
        // level's beam loop; checking it first would skip this level
        // entirely in the shared-queue descent path.
        candidate_queue.reset_exploration();
        if (candidate_queue.empty()) return;

        for (const auto& seed : candidate_queue) {
            visited.set(seed.get_vid());
        }

        const detail::DynamicLayerRange<HierarchicalGraphT> nbrs_range(hier_graph, level_id);
        detail::beam_loop_body<RouterTraitsT>(
            query_vec, nbrs_range, candidate_queue,
            visited, _dist_func, _vecs_data);
    }

private:
    const vector_array_t& _vecs_data;
    const dist_func_t&    _dist_func;

};  // class SingleLayerRouter

}   // namespace dynamic
}   // namespace cpu
}   // namespace artea
