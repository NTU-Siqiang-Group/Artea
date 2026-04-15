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
 * @Description: Single-level atom for beam-searching within one level of a
 *               dynamic::HierarchicalGraph. Exposes only the two primitives
 *               that cross-level callers actually need (seed + extend);
 *               higher-level query wrappers live in HierarchicalGraphRouter.
 */

#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

#include <artea/common/logger.hpp>

namespace artea {
namespace cpu {
namespace dynamic {

/**
 * @brief Single-level beam-search primitive for
 *        @c dynamic::HierarchicalGraph.
 *
 * The router holds only what the atom needs: the base-vector array and
 * the distance functor. It does NOT hold a hierarchical-graph reference —
 * each call takes the graph as a template-parameterized argument, so the
 * same router instance can service many graphs.
 *
 * Deliberately exposes **only** two primitives:
 *   - @c sample_entries — seed a pre-constructed candidate queue with
 *     random vertices participating at a given level.
 *   - @c beam_search    — extend a pre-seeded candidate queue by running
 *     the standard beam-search loop on one level of the graph.
 *
 * Any cross-level composition (top-down descent, batch query, etc.) lives
 * in @c HierarchicalGraphRouter on top of these two primitives.
 *
 * @tparam RouterTraitsT The router traits type (supplies candidate queue
 *                       type, visited-table type, distance functor, etc.).
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
     * @brief Seed @p candidate_queue with up to
     *        @c candidate_queue.capacity() evenly-spaced random vertices
     *        that participate at @p level_id.
     *
     * Samples directly from the top-occupied level's apex bucket
     * (@c hg.get_vids_with_highest_level(top_level_id)) — every
     * hierarchical descent starts at the apex, so restricting the pool
     * to that single bucket keeps the entry points coarse and skips any
     * cross-bucket unioning. Thread-safe via a thread-local RNG.
     */
    template <typename HierarchicalGraphT>
    auto sample_entries(
        const HierarchicalGraphT& hg,
        const vec_ele_t*          query_vec,
        std_candidate_queue_t&    candidate_queue
    ) const -> void {
        const layer_id_t top_level_id = hg.top_occupied_level_id();
        if (top_level_id == HierarchicalGraphT::unassigned_highest_level_id) {
            ARTEA_ERROR("sample_entries: hierarchy has no occupied levels");
        }
        const auto& bucket = hg.get_vids_with_highest_level(top_level_id);
        if (bucket.empty()) {
            ARTEA_ERROR("sample_entries: top-level bucket is empty");
        }

        // Acquire-fence pair to the release fence in
        // HierarchicalGraph::assign_layer: any vid we just observed in
        // the bucket is paired with a fully-published _vertex_info_table
        // entry before fetch_layer_nbrs dereferences it inside
        // beam_search.
        std::atomic_thread_fence(std::memory_order_acquire);

        const std::size_t pool_size = bucket.size();
        const std::size_t queue_cap = candidate_queue.capacity();
        const std::size_t start  = _draw_random_index(pool_size);
        const std::size_t stride =
            (pool_size <= queue_cap) ? 1 : (pool_size / queue_cap);
        const std::size_t take   = std::min(queue_cap, pool_size);

        for (std::size_t i = 0; i < take; ++i) {
            const std::size_t random_idx =
                (start + i * stride) % pool_size;
            const vertex_id_t sampled_vid = bucket[random_idx];
            const distance_t  sampled_dist =
                _dist_func(query_vec, _vecs_data.get(sampled_vid));
            candidate_queue.try_push(sampled_vid, sampled_dist);
        }
    }

    /**
     * @brief Beam search on one level of @p hg, operating in-place on a
     *        pre-seeded candidate queue.
     *
     *   1. Clears @p visited.
     *   2. Marks every seed already in the queue as visited.
     *   3. Runs the standard beam-search loop until
     *      @c candidate_queue.should_terminate().
     */
    template <typename HierarchicalGraphT>
    auto beam_search(
        const vec_ele_t*          query_vec,
        const HierarchicalGraphT& hg,
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

        visited.clear();
        for (const auto& seed : candidate_queue) {
            visited.set(seed.get_vid());
        }

        while (!candidate_queue.empty()) {
            if (candidate_queue.should_terminate()) break;
            const candidate_entry_t current = candidate_queue.pop_best_unexplored_entry();
            if (current.is_invalid()) break;

            const vertex_id_t cur_vid = current.get_vid();
            const auto nbrs_span = hg.fetch_layer_nbrs(cur_vid, level_id);
            const vertex_num_t cur_nbr_count = hg.num_valid_nbrs(cur_vid, level_id);

            for (vertex_num_t i = 0; i < cur_nbr_count; ++i) {
                const nbr_t nbr = nbrs_span[i];
                if (nbr.is_invalid()) break;
                const vertex_id_t nbr_vid = nbr.get_vid();
                if (visited.test_and_set(nbr_vid)) continue;
                const distance_t dist = _dist_func(query_vec, _vecs_data.get(nbr_vid));
                candidate_queue.try_push(nbr_vid, dist);
            }
        }
    }

private:
    /**
     * @brief Draw a single uniformly random index in @c [0, upper_bound).
     *        Uses a thread-local Mersenne Twister so concurrent callers
     *        do not contend on shared RNG state.
     */
    __attribute__((always_inline))
    static auto _draw_random_index(const std::size_t upper_bound) -> std::size_t {
        thread_local std::mt19937_64 rng(std::random_device{}());
        std::uniform_int_distribution<std::size_t> dist(0, upper_bound - 1);
        return dist(rng);
    }

    const vector_array_t& _vecs_data;
    const dist_func_t&    _dist_func;

};  // class SingleLayerRouter

}   // namespace dynamic
}   // namespace cpu
}   // namespace artea
