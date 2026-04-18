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
 * @FilePath: /Artea/include/artea/cpu/router/detail/beam_loop.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Shared inner-loop bodies for beam_search and
 *               greedy_search, decoupled from the graph storage type
 *               via a NeighborRange adapter. The 6 mode/atom router
 *               variants all reduce to one of these two functions.
 */

#pragma once

#include <utility>

#include <artea/cpu/router/data_structures/candidate_queue_concept.hpp>
#include <artea/cpu/router/neighbor_range_concept.hpp>
#include <artea/cpu/router/visited_table_concept.hpp>

namespace artea {
namespace cpu {
namespace detail {

/**
 * @brief Beam-search inner loop body. Caller is responsible for:
 *        (1) seeding @p candidate_queue,
 *        (2) calling @c reset_exploration() if reusing a queue across
 *            hierarchical layers,
 *        (3) seeding @p visited from the queue's current contents
 *            (this loop only marks newly-discovered neighbors).
 *
 * The loop pops the best unexplored candidate, expands its neighbors via
 * the @p nbrs_range adapter, computes distance for unvisited neighbors,
 * and pushes them into the queue. Terminates when the queue's
 * @c should_terminate() trips or the unexplored heap drains.
 */
template <typename RouterTraitsT, NeighborRange NeighborRangeT>
    requires CandidateQueue<typename RouterTraitsT::std_candidate_queue_t> &&
             VisitedTable  <typename RouterTraitsT::visited_table_t>
__attribute__((always_inline))
inline auto beam_loop_body(
    const typename RouterTraitsT::vec_ele_t*           query_vec,
    const NeighborRangeT&                              nbrs_range,
    typename RouterTraitsT::std_candidate_queue_t&     candidate_queue,
    typename RouterTraitsT::visited_table_t&           visited,
    const typename RouterTraitsT::dist_func_t&         dist_func,
    const typename RouterTraitsT::vector_array_t&      vecs_data
) -> void {
    using vertex_id_t       = typename RouterTraitsT::vertex_id_t;
    using distance_t        = typename RouterTraitsT::distance_t;
    using candidate_entry_t = typename RouterTraitsT::candidate_entry_t;

    while (!candidate_queue.empty()) {
        if (candidate_queue.should_terminate()) break;
        const candidate_entry_t current = candidate_queue.pop_best_unexplored_entry();
        if (current.is_invalid()) break;

        const vertex_id_t cur_vid = current.get_vid();
        for (const vertex_id_t nbr_vid : nbrs_range.of(cur_vid)) {
            if (visited.test_and_set(nbr_vid)) continue;
            const distance_t dist = dist_func(query_vec, vecs_data.get(nbr_vid));
            candidate_queue.try_push(nbr_vid, dist);
        }
    }
}

/**
 * @brief Greedy best-improvement walk. Caller is responsible for marking
 *        @p seed_vid in @p visited if subsequent iterations / layers will
 *        share the table.
 *
 * Single cursor; replaces best when a strictly closer neighbor appears.
 * Terminates at local optimum.
 */
template <typename RouterTraitsT, NeighborRange NeighborRangeT>
    requires VisitedTable<typename RouterTraitsT::visited_table_t>
__attribute__((always_inline))
inline auto greedy_loop_body(
    const typename RouterTraitsT::vec_ele_t*           query_vec,
    const NeighborRangeT&                              nbrs_range,
    typename RouterTraitsT::vertex_id_t                seed_vid,
    typename RouterTraitsT::distance_t                 seed_dist,
    typename RouterTraitsT::visited_table_t&           visited,
    const typename RouterTraitsT::dist_func_t&         dist_func,
    const typename RouterTraitsT::vector_array_t&      vecs_data
) -> std::pair<typename RouterTraitsT::vertex_id_t,
               typename RouterTraitsT::distance_t> {
    using vertex_id_t = typename RouterTraitsT::vertex_id_t;
    using distance_t  = typename RouterTraitsT::distance_t;

    vertex_id_t best_vid  = seed_vid;
    distance_t  best_dist = seed_dist;
    while (true) {
        vertex_id_t next_vid  = best_vid;
        distance_t  next_dist = best_dist;
        for (const vertex_id_t nbr_vid : nbrs_range.of(best_vid)) {
            if (visited.test_and_set(nbr_vid)) continue;
            const distance_t nbr_dist = dist_func(query_vec, vecs_data.get(nbr_vid));
            if (nbr_dist < next_dist) {
                next_dist = nbr_dist;
                next_vid  = nbr_vid;
            }
        }
        if (next_vid == best_vid) break;   // local optimum
        best_vid  = next_vid;
        best_dist = next_dist;
    }
    return {best_vid, best_dist};
}

}   // namespace detail
}   // namespace cpu
}   // namespace artea
