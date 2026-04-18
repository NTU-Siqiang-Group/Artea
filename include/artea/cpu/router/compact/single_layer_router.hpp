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
 * @FilePath: /Artea/include/artea/cpu/router/compact/single_layer_router.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Single-level beam/greedy atom over compact::HierarchicalGraph.
 *               Mirrors dynamic::SingleLayerRouter but iterates plain
 *               vertex_id_t arrays (frozen post-compaction, no locks /
 *               fences). Apex sampling lives in CandidateSampleUtils.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>

#include <artea/common/logger.hpp>
#include <artea/cpu/router/detail/beam_loop.hpp>
#include <artea/cpu/router/detail/compact_layer_range.hpp>

namespace artea {
namespace cpu {
namespace compact {

/**
 * @brief Single-level beam-search primitive for
 *        @c compact::HierarchicalGraph. Exposes @c beam_search and
 *        @c greedy_search only; apex sampling is provided by
 *        @c CandidateSampleUtils.
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
     *        were already evaluated earlier on the same walk. The caller
     *        owns the clear/reset policy; this method only seeds
     *        @p visited with @p seed_vid before the walk.
     *
     * Intended for the upper-layer descent portion of a hierarchical query
     * when the caller wants HNSW-style cheap greedy routing; the result
     * feeds the L0 beam search as a single seed.
     *
     * @param query_vec   Query vector.
     * @param hier_graph  Hierarchical graph.
     * @param level_id    Layer to walk on.
     * @param seed_vid    Starting vid (must participate at @p level_id).
     * @param seed_dist   Precomputed distance from @p query_vec to @p seed_vid.
     * @param visited     Visited table (seeded on entry; caller resets).
     * @return @c (best_vid, best_dist) reached at this level.
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
        const detail::CompactLayerRange<HierarchicalGraphT> nbrs_range(hier_graph, level_id);
        return detail::greedy_loop_body<RouterTraitsT>(
            query_vec, nbrs_range, seed_vid, seed_dist,
            visited, _dist_func, _vecs_data);
    }

    /**
     * @brief Beam search on one level of @p hier_graph, operating in-place on a
     *        pre-seeded candidate queue. Does NOT clear @p visited —
     *        caller owns clear lifecycle so visited can be shared
     *        across hierarchical layers. Only marks the current queue
     *        seeds, which is idempotent for carry-over seeds.
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
        // Repopulate _unexplored_set from _top_candidates BEFORE the
        // empty check. candidate_queue.empty() inspects _unexplored_set,
        // which is fully drained after the previous level's beam loop;
        // checking it first would skip this level entirely under shared-
        // queue hierarchical descent, leaving every seed from the upper
        // level unexpanded at this level.
        candidate_queue.reset_exploration();
        if (candidate_queue.empty()) return;

        for (const auto& seed : candidate_queue) {
            visited.set(seed.get_vid());
        }

        const detail::CompactLayerRange<HierarchicalGraphT> nbrs_range(hier_graph, level_id);
        detail::beam_loop_body<RouterTraitsT>(
            query_vec, nbrs_range, candidate_queue,
            visited, _dist_func, _vecs_data);
    }

private:
    const vector_array_t& _vecs_data;
    const dist_func_t&    _dist_func;

};  // class SingleLayerRouter

}   // namespace compact
}   // namespace cpu
}   // namespace artea
