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
 * @FilePath: /Artea/include/artea/cpu/router/single_layer_router.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Unified single-level beam/greedy atom. Graph-storage-
 *               agnostic: callers pass a NeighborRange adapter that
 *               yields vertex_id_t and stops at a sentinel. Replaces
 *               the per-mode compact::SingleLayerRouter and
 *               dynamic::SingleLayerRouter.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>

#include <artea/common/logger.hpp>
#include <artea/cpu/router/detail/beam_loop.hpp>
#include <artea/cpu/router/neighbor_range_concept.hpp>

namespace artea {
namespace cpu {

/**
 * @brief Single-level beam-search primitive parameterized on a
 *        @c NeighborRange adapter. Apex sampling lives in
 *        @c CandidateSampleUtils so callers can swap policies without
 *        dragging RNG state into the router.
 *
 * @tparam RouterTraitsT The router traits type.
 */
template <typename RouterTraitsT>
class SingleLayerRouter {

    using vertex_num_t          = typename RouterTraitsT::vertex_num_t;
    using vertex_id_t           = typename RouterTraitsT::vertex_id_t;
    using vec_ele_t             = typename RouterTraitsT::vec_ele_t;
    using distance_t            = typename RouterTraitsT::distance_t;
    using dist_func_t           = typename RouterTraitsT::dist_func_t;
    using vector_array_t        = typename RouterTraitsT::vector_array_t;
    using std_candidate_queue_t = typename RouterTraitsT::std_candidate_queue_t;
    using visited_table_t       = typename RouterTraitsT::visited_table_t;

public:
    SingleLayerRouter(
        const vector_array_t& base_vecs,
        const dist_func_t&    dist_func
    ) :
        _vecs_data(base_vecs),
        _dist_func(dist_func) {}

    /**
     * @brief Greedy best-improvement walk over @p nbrs_range. Uses
     *        @p visited to skip distance computation for vids already
     *        evaluated on this walk. The caller owns the clear/reset
     *        policy; this method only seeds @p visited with @p seed_vid.
     *
     * Intended for the upper-layer descent portion of a hierarchical
     * query when the caller wants HNSW-style cheap greedy routing; the
     * result feeds the L0 beam search as a single seed.
     */
    template <NeighborRange NeighborRangeT>
    __attribute__((always_inline))
    auto greedy_search(
        const vec_ele_t*       query_vec,
        const NeighborRangeT&  nbrs_range,
        vertex_id_t            seed_vid,
        distance_t             seed_dist,
        visited_table_t&       visited
    ) const -> std::pair<vertex_id_t, distance_t> {
        visited.set(seed_vid);
        return detail::greedy_loop_body<RouterTraitsT>(
            query_vec, nbrs_range, seed_vid, seed_dist,
            visited, _dist_func, _vecs_data);
    }

    /**
     * @brief Beam search over @p nbrs_range, operating in-place on a
     *        pre-seeded candidate queue. Does NOT clear @p visited —
     *        caller owns clear lifecycle so visited can be shared
     *        across hierarchical layers (and across the insert descent
     *        + select phases). Only marks the current queue seeds,
     *        which is idempotent for carry-over seeds.
     *
     * @c reset_exploration() MUST run BEFORE the @c empty() check:
     * @c empty() inspects the unexplored heap, which is fully drained
     * after the previous level's beam loop; checking it first would
     * skip this level entirely under shared-queue hierarchical descent.
     */
    template <NeighborRange NeighborRangeT>
    __attribute__((always_inline))
    auto beam_search(
        const vec_ele_t*       query_vec,
        const NeighborRangeT&  nbrs_range,
        std_candidate_queue_t& candidate_queue,
        visited_table_t&       visited
    ) const -> void {
        candidate_queue.reset_exploration();
        if (candidate_queue.empty()) return;

        for (const auto& seed : candidate_queue) {
            visited.set(seed.get_vid());
        }

        detail::beam_loop_body<RouterTraitsT>(
            query_vec, nbrs_range, candidate_queue,
            visited, _dist_func, _vecs_data);
    }

private:
    const vector_array_t& _vecs_data;
    const dist_func_t&    _dist_func;

};  // class SingleLayerRouter

}   // namespace cpu
}   // namespace artea
