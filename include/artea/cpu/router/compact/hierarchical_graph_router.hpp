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
 * @FilePath: /Artea/include/artea/cpu/router/compact/hierarchical_graph_router.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Multi-level query router over compact::HierarchicalGraph.
 *               Composes a compact::SingleLayerRouter for the per-level
 *               atom; cross-level composition is a straight top-down
 *               descent (no inter-layer candidate translation because
 *               every vertex has one global vid that is valid at every
 *               level it participates in). Exposes query / batch_query
 *               for the read-only query path.
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <artea/common/logger.hpp>
#include <artea/cpu/router/compact/single_layer_router.hpp>

namespace artea {
namespace cpu {
namespace compact {

/**
 * @brief Multi-level query-only router over @c compact::HierarchicalGraph.
 *
 * Exposes:
 *   - @c query(query_vec, hg) — top-down beam search, returns top-k.
 *   - @c batch_query(query_vecs, hg) — parallel version via TBB.
 *
 * Because compact::HierarchicalGraph is read-only post-compaction there
 * is no build-time passthrough (unlike the dynamic counterpart used by
 * @c stacked_rgraph::IndexFactory).
 *
 * @tparam RouterTraitsT The router traits type.
 */
template <typename RouterTraitsT>
class HierarchicalGraphRouter :
    public RouterTraitsT::template vector_router_t<HierarchicalGraphRouter<RouterTraitsT>>
{
    using vertex_num_t            = typename RouterTraitsT::vertex_num_t;
    using vertex_id_t             = typename RouterTraitsT::vertex_id_t;
    using layer_id_t              = typename RouterTraitsT::layer_id_t;
    using vec_ele_t               = typename RouterTraitsT::vec_ele_t;
    using distance_t              = typename RouterTraitsT::distance_t;
    using dist_func_t             = typename RouterTraitsT::dist_func_t;
    using vector_array_t          = typename RouterTraitsT::vector_array_t;
    using query_vecs_t            = typename RouterTraitsT::query_vecs_t;
    using candidate_entry_t       = typename RouterTraitsT::candidate_entry_t;
    using std_candidate_queue_t   = typename RouterTraitsT::std_candidate_queue_t;
    using visited_table_t         = typename RouterTraitsT::visited_table_t;
    using visited_table_pool_t    = typename RouterTraitsT::visited_table_pool_t;
    using knn_results_t           = typename RouterTraitsT::knn_results_t;
    using base_class_t            =
        typename RouterTraitsT::template vector_router_t<HierarchicalGraphRouter<RouterTraitsT>>;

    using single_layer_router_t   = SingleLayerRouter<RouterTraitsT>;

    static constexpr vertex_id_t invalid_vertex_id = RouterTraitsT::invalid_vertex_id;

public:
    /**
     * @brief Construct a compact HierarchicalGraphRouter.
     *
     * @param base_vecs             Base dataset (indexed by vid).
     * @param dist_func             Distance functor.
     * @param topk                  Top-k returned by @c query and
     *                              @c batch_query.
     * @param search_nn_qs          Beam width used at every level during
     *                              the top-down query descent.
     * @param candidate_queue_size  Candidate-queue capacity (should be
     *                              >= topk).
     */
    HierarchicalGraphRouter(
        const vector_array_t& base_vecs,
        const dist_func_t&    dist_func,
        const uint32_t        topk,
        const vertex_num_t    search_nn_qs,
        const vertex_num_t    candidate_queue_size = 16
    ) :
        base_class_t(base_vecs, dist_func, topk),
        _single_layer_router(base_vecs, dist_func),
        _search_nn_qs(search_nn_qs),
        _candidate_queue_size(candidate_queue_size),
        _visited_table_pool(base_vecs.get_num_vecs())
    {
        if (search_nn_qs == 0) {
            ARTEA_ERROR("search_nn_qs must be >= 1");
        }
        if (candidate_queue_size < topk) {
            ARTEA_ERROR(fmt::format(
                "candidate_queue_size ({}) must be >= topk ({})",
                candidate_queue_size, topk));
        }
    }

    /** @brief Warm up the visited-table pool (query hot path only). */
    auto initialize() -> void {
        _visited_table_pool.warmup();
    }

    // ================================================================
    //   Query path
    // ================================================================

    /**
     * @brief Top-down hierarchical search.
     *
     * Both branches start from the compactor-precomputed
     * @c hg.entry_point_vid() (the top-bucket centroid's nearest
     * neighbor) — no runtime sampling.
     *
     * @tparam UpperLevelBeamSearch
     *   - @c true (default): legacy path — allocate one candidate
     *     queue of @c _candidate_queue_size, seed it with the entry
     *     point, then run @c beam_search on every level from top
     *     down to L0 with the same queue.
     *   - @c false: HNSW-style split. Upper levels (top..1) run
     *     @c greedy_search with no queue at all, propagating a single
     *     best (vid, distance) cursor starting at the entry point.
     *     At L0 we allocate the candidate queue, seed it with the
     *     L1 greedy-best cursor, and run @c beam_search once.
     *
     * Either way the final top-k is extracted from the L0 candidate
     * queue.
     */
    template <bool UpperLevelBeamSearch = true, typename HierarchicalGraphT>
    auto query(
        const vec_ele_t*          query_vec,
        const HierarchicalGraphT& hg
    ) const -> knn_results_t {
        const layer_id_t top_level_id = hg.top_occupied_level_id();
        // This is impossible
        // if (top_level_id == HierarchicalGraphT::unassigned_highest_level_id) {
        //     return knn_results_t{};
        // }

        auto& visited = _visited_table_pool.acquire();

        // Every query starts from the cached entry point (top-bucket
        // centroid's nearest neighbor), precomputed by the compactor.
        // No runtime sampling.
        const vertex_id_t entry_vid = hg.entry_point_vid();
        const distance_t  entry_dist =
            this->_dist_func(query_vec, this->_vecs_data.get(entry_vid));

        if constexpr (UpperLevelBeamSearch) {
            // ---- Legacy: beam search at every level ----
            std_candidate_queue_t candidate_queue(
                static_cast<std::size_t>(_candidate_queue_size));
            candidate_queue.try_push(entry_vid, entry_dist);

            for (layer_id_t cur_level_id = top_level_id; ; --cur_level_id) {
                _single_layer_router.beam_search(
                    query_vec, hg, cur_level_id, candidate_queue, visited);
                if (cur_level_id == 0) break;
            }

            const std::size_t k = std::min<std::size_t>(
                static_cast<std::size_t>(this->_topk),
                candidate_queue.get_result_size());
            visited.clear();
            if (k == 0) return knn_results_t{};
            return candidate_queue.extract_results(k);
        } else {
            // ---- Greedy upper levels + beam at L0 ----
            vertex_id_t cursor_vid  = entry_vid;
            distance_t  cursor_dist = entry_dist;

            // Greedy descent from top_level_id down to L1 (no queues).
            // If top_level_id == 0 this loop is skipped and the entry
            // point is fed straight into the L0 beam below.
            for (layer_id_t cur_level_id = top_level_id;
                 cur_level_id >= 1;
                 --cur_level_id)
            {
                std::tie(cursor_vid, cursor_dist) =
                    _single_layer_router.greedy_search(
                        query_vec, hg, cur_level_id,
                        cursor_vid, cursor_dist);
                if (cur_level_id == 1) break;
            }

            // L0: allocate the candidate queue, seed it with the L1
            // greedy-best cursor, and run beam search once.
            std_candidate_queue_t candidate_queue(
                static_cast<std::size_t>(_candidate_queue_size));
            candidate_queue.try_push(cursor_vid, cursor_dist);
            _single_layer_router.beam_search(
                query_vec, hg, /*level_id=*/layer_id_t{0},
                candidate_queue, visited);

            const std::size_t k = std::min<std::size_t>(
                static_cast<std::size_t>(this->_topk),
                candidate_queue.get_result_size());
            visited.clear();
            if (k == 0) return knn_results_t{};
            return candidate_queue.extract_results(k);
        }
    }

    /**
     * @brief L0-only search baseline: skip the hierarchy entirely and
     *        run a single beam search on the base layer starting from
     *        the compactor-precomputed @c entry_point_vid. Provided
     *        so callers can measure how much the hierarchical descent
     *        actually buys them vs. flat-graph routing on the same
     *        entry point.
     */
    template <typename HierarchicalGraphT>
    auto query_l0_only(
        const vec_ele_t*          query_vec,
        const HierarchicalGraphT& hg
    ) const -> knn_results_t {
        auto& visited = _visited_table_pool.acquire();

        const vertex_id_t entry_vid = hg.entry_point_vid();
        const distance_t  entry_dist =
            this->_dist_func(query_vec, this->_vecs_data.get(entry_vid));

        std_candidate_queue_t candidate_queue(
            static_cast<std::size_t>(_candidate_queue_size));
        candidate_queue.try_push(entry_vid, entry_dist);
        _single_layer_router.beam_search(
            query_vec, hg, /*level_id=*/layer_id_t{0},
            candidate_queue, visited);

        const std::size_t k = std::min<std::size_t>(
            static_cast<std::size_t>(this->_topk),
            candidate_queue.get_result_size());
        visited.clear();
        if (k == 0) return knn_results_t{};
        return candidate_queue.extract_results(k);
    }

    /**
     * @brief Parallel batch queries via TBB. Each worker acquires its
     *        own visited table from the pool. The
     *        @p UpperLevelBeamSearch template switch is forwarded to
     *        @c query verbatim.
     */
    template <bool UpperLevelBeamSearch = true, typename HierarchicalGraphT>
    auto batch_query(
        const query_vecs_t&       query_vecs,
        const HierarchicalGraphT& hg
    ) const -> knn_results_t {
        const vertex_num_t num_queries = query_vecs.get_num_vecs();
        const uint32_t     k           = this->_topk;

        knn_results_t results(num_queries * k);

        tbb::parallel_for(
            tbb::blocked_range<vertex_num_t>(0, num_queries),
            [&](const tbb::blocked_range<vertex_num_t>& r) {
                for (vertex_num_t i = r.begin(); i != r.end(); ++i) {
                    const vec_ele_t* q_vec = query_vecs.get(i);
                    auto topk_results =
                        this->template query<UpperLevelBeamSearch>(q_vec, hg);
                    const std::size_t n = topk_results.size();
                    std::copy(topk_results.begin(), topk_results.end(),
                              results.begin() + i * k);
                    for (std::size_t j = n; j < k; ++j) {
                        results[i * k + j] =
                            candidate_entry_t::make_invalid_entry();
                    }
                }
            }
        );

        return results;
    }

    /**
     * @brief Parallel batch form of @c query_l0_only. Same per-worker
     *        visited-table pooling as @c batch_query.
     */
    template <typename HierarchicalGraphT>
    auto batch_query_l0_only(
        const query_vecs_t&       query_vecs,
        const HierarchicalGraphT& hg
    ) const -> knn_results_t {
        const vertex_num_t num_queries = query_vecs.get_num_vecs();
        const uint32_t     k           = this->_topk;

        knn_results_t results(num_queries * k);

        tbb::parallel_for(
            tbb::blocked_range<vertex_num_t>(0, num_queries),
            [&](const tbb::blocked_range<vertex_num_t>& r) {
                for (vertex_num_t i = r.begin(); i != r.end(); ++i) {
                    const vec_ele_t* q_vec = query_vecs.get(i);
                    auto topk_results = this->query_l0_only(q_vec, hg);
                    const std::size_t n = topk_results.size();
                    std::copy(topk_results.begin(), topk_results.end(),
                              results.begin() + i * k);
                    for (std::size_t j = n; j < k; ++j) {
                        results[i * k + j] =
                            candidate_entry_t::make_invalid_entry();
                    }
                }
            }
        );

        return results;
    }

private:
    single_layer_router_t        _single_layer_router;
    vertex_num_t                 _search_nn_qs;
    vertex_num_t                 _candidate_queue_size;
    mutable visited_table_pool_t _visited_table_pool;

};  // class HierarchicalGraphRouter

}   // namespace compact
}   // namespace cpu
}   // namespace artea
