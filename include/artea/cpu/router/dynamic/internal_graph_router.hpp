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
 * @FilePath: /Artea/include/artea/cpu/router/internal_graph_router.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Single-layer router operating directly on an InternalGraph.
 *               Uses a inbr-backed StdCandidateQueue so each candidate
 *               carries both @c base_vid (for coordinate lookup / inter-layer
 *               descent) and @c layer_vid (for the visited table and CSR
 *               traversal) without a side-channel hash map.
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <utility>
#include <vector>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <artea/common/logger.hpp>
#include <artea/cpu/utils/parallel.hpp>
#include <artea/cpu/router/data_structures/candidate_queue_concept.hpp>
#include <artea/cpu/router/visited_table_concept.hpp>

namespace artea {
namespace cpu {
namespace dynamic {

/**
 * @brief Single-layer proximity-graph router operating on an InternalGraph.
 *
 * This router is the counterpart of @c BottomGraphRouter<...::compact_mode>
 * for the dynamic-insertion index family. It carries both @c base_vid and
 * @c layer_vid through the beam-search candidate queue (via
 * @c CandidateEntry), so callers — including
 * @c stacked_rgraph::IndexFactory on its build-time path — can recover both
 * identifiers directly from extracted results without a side-channel
 * @c std::unordered_map<layer_vid, base_vid>.
 *
 * The graph is NOT stored by reference. Every public method takes the
 * @c InternalGraphT to operate on as a template parameter at call time,
 * matching the pattern used by @c BottomGraphRouter<...::dynamic_mode>.
 *
 * @tparam RouterTraitsT The router traits type.
 */
template <typename RouterTraitsT>
class InternalGraphRouter :
    public RouterTraitsT::template vector_router_t<InternalGraphRouter<RouterTraitsT>>
{

    using vertex_num_t            = typename RouterTraitsT::vertex_num_t;
    using vertex_id_t             = typename RouterTraitsT::vertex_id_t;
    using vec_ele_t               = typename RouterTraitsT::vec_ele_t;
    using distance_t              = typename RouterTraitsT::distance_t;
    using dist_func_t             = typename RouterTraitsT::dist_func_t;
    using vector_array_t          = typename RouterTraitsT::vector_array_t;
    using query_vecs_t            = typename RouterTraitsT::query_vecs_t;
    using inbr_t                  = typename RouterTraitsT::inbr_t;
    using candidate_entry_t       = typename RouterTraitsT::candidate_entry_t;
    using std_candidate_queue_t   = typename RouterTraitsT::std_candidate_queue_t;
    using visited_table_t         = typename RouterTraitsT::visited_table_t;
    using visited_table_pool_t    = typename RouterTraitsT::visited_table_pool_t;
    using knn_results_t           = typename RouterTraitsT::knn_results_t;
    using base_class_t            = typename RouterTraitsT::template vector_router_t<InternalGraphRouter<RouterTraitsT>>;

    static constexpr vertex_id_t invalid_vertex_id = RouterTraitsT::invalid_vertex_id;

public:
    /**
     * @brief Construct an InternalGraphRouter.
     *
     * @param base_vecs             Base dataset. Coordinates are fetched via
     *                              @c base_vecs.get(base_vid) inside
     *                              @c beam_search and its callers.
     * @param dist_func             Distance function functor.
     * @param topk                  Number of nearest neighbors returned by
     *                              @c query() / @c batch_query() / @c beam_search().
     * @param candidate_queue_size  Beam width used by the high-level
     *                              @c beam_search() / @c query() entry
     *                              points. The low-level
     *                              @c beam_search() takes its width
     *                              as an explicit argument.
     */
    InternalGraphRouter(
        const vector_array_t& base_vecs,
        const dist_func_t& dist_func,
        const uint32_t topk,
        const vertex_num_t candidate_queue_size = 16
    ) : base_class_t(base_vecs, dist_func, topk),
        _candidate_queue_size(candidate_queue_size),
        _visited_table_pool(base_vecs.get_num_vecs())
    {
        if (candidate_queue_size < topk) {
            ARTEA_ERROR(fmt::format(
                "candidate_queue_size ({}) must be >= topk ({})",
                candidate_queue_size, topk
            ));
        }
    }

    /**
     * @brief Warm up the visited-table pool and the random-sequence generator.
     *
     * Only needed before the FIRST query on the hot path. The build-time
     * primitive (@c beam_search) does NOT require this — it takes a
     * caller-owned visited table and never samples random entries.
     */
    auto initialize() -> void {
        _visited_table_pool.warmup();
    }

    // ================================================================
    //   Build-time / atomic primitive
    // ================================================================

    /**
     * @brief Beam search on a single InternalGraph layer, operating
     *        in-place on a pre-seeded candidate queue.
     *
     * The caller prepares @p candidate_queue with seed entries (via try_push or
     * sample_entries) before invoking this method. The method
     * clears @p visited, marks every seed entry as visited, then runs
     * the standard beam-search loop until termination.
     *
     * Torn-read guards:
     *   - Neighbor @c layer_vid is clamped against the layer size.
     *   - Neighbor count is clamped against @c layer.max_nbr_size().
     *
     * @tparam InternalGraphT The internal-graph type.
     * @param query_vec   Pointer to the query vector data.
     * @param layer       The InternalGraph layer to search.
     * @param candidate_queue          Pre-seeded candidate queue (modified in-place).
     * @param visited     Caller-owned visited table. Cleared on entry.
     */
    template <typename InternalGraphT>
    auto beam_search(
        const vec_ele_t* query_vec,
        const InternalGraphT& layer,
        std_candidate_queue_t& candidate_queue,
        visited_table_t& visited
    ) const -> void {
        const vertex_num_t layer_size = layer.get_num_vertices();
        if (layer_size == 0 || candidate_queue.empty()) return;

        const vertex_num_t max_nbr = layer.max_nbr_size();

        // Mark all in-bounds seed entries as visited. Out-of-bounds
        // seeds (stale from concurrent extension) are marked too; the
        // guard below prevents us from dereferencing them.
        visited.clear();
        for (const auto& e : candidate_queue) {
            const vertex_id_t lv = e.get_layer_vid();
            if (lv < layer_size) visited.set(lv);
        }

        // Standard beam-search loop.
        while (!candidate_queue.empty()) {
            if (candidate_queue.should_terminate()) break;
            const candidate_entry_t current = candidate_queue.pop_best_unexplored_entry();
            if (current.is_invalid()) break;

            const vertex_id_t cur_lv = current.get_layer_vid();
            if (cur_lv >= layer_size) continue;  // stale seed, skip safely

            const auto block = layer.fetch_nbrs(cur_lv);
            const uint64_t raw_count = layer.num_valid_nbrs(cur_lv);
            const uint64_t count = std::min<uint64_t>(raw_count, max_nbr);
            for (uint64_t i = 0; i < count; ++i) {
                const inbr_t nbr = block[1 + i];
                if (nbr.get_base_vid() == invalid_vertex_id) continue;
                if (nbr.get_level_vid() >= layer_size) continue;  // torn read
                if (visited.test_and_set(nbr.get_level_vid())) continue;
                const distance_t dist =
                    this->_dist_func(query_vec, this->_vecs_data.get(nbr.get_base_vid()));
                candidate_queue.try_push(nbr.get_base_vid(), nbr.get_level_vid(), dist);
            }
        }
    }

    // ================================================================
    //   High-level query API
    // ================================================================

    /**
     * @brief Beam search for the top-k nearest vertices on a single layer,
     *        seeded with uniformly random entries drawn from @p layer.
     */
    template <typename InternalGraphT>
    auto beam_search(
        const vec_ele_t* query_vec,
        const InternalGraphT& layer
    ) const -> knn_results_t {
        const vertex_num_t layer_size = layer.get_num_vertices();
        if (layer_size == 0) return {};

        auto& visited = _visited_table_pool.acquire();

        std_candidate_queue_t candidate_queue(static_cast<std::size_t>(_candidate_queue_size));
        sample_entries(layer, query_vec, candidate_queue);
        beam_search(query_vec, layer, candidate_queue, visited);

        visited.clear();
        const std::size_t topk = std::min(
            static_cast<std::size_t>(this->_topk), candidate_queue.get_result_size());
        if (topk == 0) return {};
        return candidate_queue.extract_results(topk);
    }

    /**
     * @brief Greedy (best-first) hill-climb search on a single layer.
     *
     * Starts from a single random vertex, then repeatedly advances to the
     * strictly-closest neighbor of the current best until no neighbor can
     * improve the current distance. Returns the final
     * @c candidate_entry_t (base_vid + layer_vid + distance).
     *
     * This is the per-layer primitive that a hierarchical greedy descent
     * composes across layers.
     */
    template <typename InternalGraphT>
    auto greedy_search(
        const vec_ele_t* query_vec,
        const InternalGraphT& layer
    ) const -> candidate_entry_t {
        const vertex_num_t layer_size = layer.get_num_vertices();
        if (layer_size == 0) return candidate_entry_t::make_invalid_entry();

        // Draw a single random layer_vid directly in [0, layer_size).
        const vertex_id_t start_lv = _draw_random_vid(layer_size);
        const vertex_id_t start_bv = layer.get_base_vid(start_lv);
        const distance_t  start_dist =
            this->_dist_func(query_vec, this->_vecs_data.get(start_bv));

        auto& visited = _visited_table_pool.acquire();
        auto result = _greedy_from(
            query_vec, layer,
            candidate_entry_t(start_bv, start_lv, start_dist),
            visited);
        visited.clear();
        return result;
    }

    /**
     * @brief Greedy hill-climb starting from a given entry point. Exposed
     *        so the hierarchical router can call it directly without
     *        another random draw. The caller owns @p visited (cleared on
     *        entry; may contain marks from prior calls).
     */
    template <typename InternalGraphT>
    auto greedy_search(
        const vec_ele_t* query_vec,
        const InternalGraphT& layer,
        const candidate_entry_t& start,
        visited_table_t& visited
    ) const -> candidate_entry_t {
        return _greedy_from(query_vec, layer, start, visited);
    }

    /**
     * @brief Query the top-k nearest vertices on a single layer.
     *        Thin wrapper over @c beam_search().
     */
    template <typename InternalGraphT>
    __attribute__((always_inline))
    auto query(
        const vec_ele_t* query_vec,
        const InternalGraphT& layer
    ) const -> knn_results_t {
        return beam_search(query_vec, layer);
    }

    /**
     * @brief Batch queries over a single layer. Parallelized over queries
     *        with TBB; each worker acquires its own visited table from the
     *        pool.
     */
    template <typename InternalGraphT>
    auto batch_query(
        const query_vecs_t& query_vecs,
        const InternalGraphT& layer
    ) const -> knn_results_t {
        const vertex_num_t num_queries = query_vecs.get_num_vecs();
        const uint32_t K = this->_topk;

        knn_results_t results(num_queries * K);

        tbb::parallel_for(
            tbb::blocked_range<vertex_num_t>(0, num_queries),
            [&](const tbb::blocked_range<vertex_num_t>& r) {
                for (vertex_num_t i = r.begin(); i != r.end(); ++i) {
                    const vec_ele_t* q_vec = query_vecs.get(i);
                    auto topk_results = this->query(q_vec, layer);
                    // Zero-pad short result rows (query returns up to K).
                    const std::size_t n = topk_results.size();
                    std::copy(topk_results.begin(), topk_results.end(),
                              results.begin() + i * K);
                    for (std::size_t j = n; j < K; ++j) {
                        results[i * K + j] = candidate_entry_t::make_invalid_entry();
                    }
                }
            }
        );

        return results;
    }

    // ================================================================
    //   Public sampling primitive (shared with HierarchicalGraphRouter
    //   and stacked_rgraph::IndexFactory)
    // ================================================================

    /**
     * @brief Sample up to @p candidate_queue.capacity() evenly-spaced entries from
     *        @p layer, compute distances to @p query_vec, and push them
     *        directly into @p candidate_queue.
     *
     * Uses a single random starting offset and a fixed stride
     * (layer_size / queue_size) to pick queue_size distinct vertices
     * without deduplication overhead. If @p layer has fewer vertices
     * than the queue capacity, every vertex is taken.
     *
     * Thread-safe: the starting offset is drawn from the thread-local
     * MKL random sequence.
     */
    template <typename InternalGraphT>
    auto sample_entries(
        const InternalGraphT& layer,
        const vec_ele_t* query_vec,
        std_candidate_queue_t& candidate_queue
    ) const -> void {
        const vertex_num_t layer_size = layer.get_num_vertices();
        if (layer_size == 0) return;

        const std::size_t queue_size = candidate_queue.capacity();

        if (static_cast<std::size_t>(layer_size) <= queue_size) {
            for (vertex_num_t lv = 0; lv < layer_size; ++lv) {
                const vertex_id_t bv = layer.get_base_vid(lv);
                const distance_t dist =
                    this->_dist_func(query_vec, this->_vecs_data.get(bv));
                candidate_queue.try_push(bv, lv, dist);
            }
            return;
        }

        // Draw a random starting offset.
        const vertex_num_t start = _draw_random_vid(layer_size);
        const vertex_num_t interval =
            layer_size / static_cast<vertex_num_t>(queue_size);

        for (std::size_t i = 0; i < queue_size; ++i) {
            const vertex_id_t lv =
                (start + static_cast<vertex_num_t>(i) * interval) % layer_size;
            const vertex_id_t bv = layer.get_base_vid(lv);
            const distance_t dist =
                this->_dist_func(query_vec, this->_vecs_data.get(bv));
            candidate_queue.try_push(bv, lv, dist);
        }
    }

private:
    // ---- Helpers ----

    /**
     * @brief Greedy hill-climb loop from a given entry.
     */
    template <typename InternalGraphT>
    auto _greedy_from(
        const vec_ele_t* query_vec,
        const InternalGraphT& layer,
        candidate_entry_t current,
        visited_table_t& visited
    ) const -> candidate_entry_t {
        const vertex_num_t layer_size = layer.get_num_vertices();
        const vertex_num_t max_nbr = layer.max_nbr_size();

        visited.clear();
        if (current.get_layer_vid() < layer_size) {
            visited.set(current.get_layer_vid());
        }

        while (true) {
            const vertex_id_t cur_lv = current.get_layer_vid();
            if (cur_lv >= layer_size) break;  // stale/torn

            const auto block = layer.fetch_nbrs(cur_lv);
            const uint64_t raw_count = layer.num_valid_nbrs(cur_lv);
            const uint64_t count = std::min<uint64_t>(raw_count, max_nbr);

            candidate_entry_t best = current;
            for (uint64_t i = 0; i < count; ++i) {
                const inbr_t nbr = block[1 + i];
                if (nbr.get_base_vid() == invalid_vertex_id) continue;
                if (nbr.get_level_vid() >= layer_size) continue;  // torn read
                if (visited.test_and_set(nbr.get_level_vid())) continue;
                const distance_t dist =
                    this->_dist_func(query_vec, this->_vecs_data.get(nbr.get_base_vid()));
                if (dist < best.get_distance()) {
                    best = candidate_entry_t(nbr.get_base_vid(), nbr.get_level_vid(), dist);
                }
            }

            if (best.get_layer_vid() == current.get_layer_vid()) {
                // No improvement; terminate.
                break;
            }
            current = best;
        }

        return current;
    }

    /**
     * @brief Draw a single uniformly random vertex_id in [0, upper_bound).
     *        Uses a thread-local @c std::mt19937 so concurrent callers do
     *        not contend on shared state.
     */
    __attribute__((always_inline))
    static auto _draw_random_vid(const vertex_num_t upper_bound) -> vertex_id_t {
        thread_local std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<vertex_id_t> dist(
            0, static_cast<vertex_id_t>(upper_bound) - 1);
        return dist(rng);
    }

    // ---- Members ----

    /** @brief Beam width used by high-level query entry points. */
    vertex_num_t _candidate_queue_size = 0;

    /** @brief Pool of thread-local visited bitmaps for parallel beam search. */
    mutable visited_table_pool_t _visited_table_pool;

};  // class InternalGraphRouter

}   // namespace dynamic
}   // namespace cpu
}   // namespace artea
