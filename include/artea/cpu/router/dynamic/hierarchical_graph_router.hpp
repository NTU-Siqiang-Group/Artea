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
 * @FilePath: /Artea/include/artea/cpu/router/dynamic/hierarchical_graph_router.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Multi-level router over dynamic::HierarchicalGraph.
 *               Composes a SingleLayerRouter for the per-level atom;
 *               cross-level composition is a straight top-down descent
 *               (no inter-layer candidate translation because every
 *               vertex has one global vid that is valid at every level
 *               it participates in).
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <artea/common/logger.hpp>
#include <artea/cpu/router/dynamic/single_layer_router.hpp>

namespace artea {
namespace cpu {
namespace dynamic {

/**
 * @brief Multi-level proximity-graph router over
 *        @c dynamic::HierarchicalGraph.
 *
 * Provides:
 *   - **Build-time passthroughs** — @c beam_search(query, hier_graph, level_id,
 *     queue, visited) and @c sample_entries(hier_graph, query, queue) that
 *     simply delegate to the composed @c SingleLayerRouter. Used by
 *     @c stacked_rgraph::IndexFactory on the insertion hot path.
 *   - **Query path** — @c beam_search(query, hier_graph) runs the full top-down
 *     descent. Because every vid is valid at every level it participates
 *     in, the candidate queue carries through each level unchanged.
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
    using candidate_sample_utils_t = typename RouterTraitsT::candidate_sample_utils_t;
    using base_class_t            =
        typename RouterTraitsT::template vector_router_t<HierarchicalGraphRouter<RouterTraitsT>>;

    using single_layer_router_t   = SingleLayerRouter<RouterTraitsT>;

    static constexpr vertex_id_t invalid_vertex_id = RouterTraitsT::invalid_vertex_id;

public:
    /**
     * @brief Construct a HierarchicalGraphRouter.
     *
     * @param base_vecs             Base dataset (indexed by vid).
     * @param dist_func             Distance functor.
     * @param topk                  Top-k returned by @c query and
     *                              @c batch_query.
     * @param candidate_queue_size  Candidate-queue capacity (should be
     *                              >= topk).
     */
    HierarchicalGraphRouter(
        const vector_array_t& base_vecs,
        const dist_func_t&    dist_func,
        const uint32_t        topk,
        const vertex_num_t    candidate_queue_size = 16
    ) :
        base_class_t(base_vecs, dist_func, topk),
        _single_layer_router(base_vecs, dist_func),
        _candidate_queue_size(candidate_queue_size),
        _visited_table_pool(base_vecs.get_num_vecs())
    {
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
    //   Build-time passthrough (beam_search only; apex sampling now
    //   flows through CandidateSampleUtils directly at the call site).
    // ================================================================

    template <typename HierarchicalGraphT>
    __attribute__((always_inline))
    auto beam_search(
        const vec_ele_t*          query_vec,
        const HierarchicalGraphT& hier_graph,
        const layer_id_t          level_id,
        std_candidate_queue_t&    candidate_queue,
        visited_table_t&          visited
    ) const -> void {
        _single_layer_router.beam_search(
            query_vec, hier_graph, level_id, candidate_queue, visited);
    }

    // ================================================================
    //   Query path
    // ================================================================

    /**
     * @brief Top-down hierarchical search.
     *
     * @tparam UpperLevelBeamSearch
     *   - @c true (default): legacy path — one candidate queue of
     *     @c _candidate_queue_size seeded via @c sample_entries at the
     *     top level, then beam_search on the same queue from top..L0.
     *   - @c false: HNSW-style split. Upper levels (top..1) run
     *     @c greedy_search queue-free from a single sampled seed; at L0
     *     we allocate the candidate queue, seed it with the L1 greedy
     *     cursor, and run beam_search once.
     */
    template <bool UpperLevelBeamSearch = true, typename HierarchicalGraphT>
    auto query(
        const vec_ele_t*          query_vec,
        const HierarchicalGraphT& hier_graph
    ) const -> knn_results_t {
        const layer_id_t top_level_id = hier_graph.top_occupied_level_id();
        if (top_level_id == HierarchicalGraphT::unassigned_highest_level_id) {
            return knn_results_t{};
        }

        auto& visited = _visited_table_pool.acquire();

        if constexpr (UpperLevelBeamSearch) {
            // ---- Legacy: beam search at every level ----
            std_candidate_queue_t candidate_queue(static_cast<std::size_t>(_candidate_queue_size));
            candidate_sample_utils_t::sample_entries(
                this->_vecs_data, this->_dist_func, hier_graph, query_vec, candidate_queue);

            for (layer_id_t cur_level_id = top_level_id; ; --cur_level_id) {
                _single_layer_router.beam_search(query_vec, hier_graph, cur_level_id, candidate_queue, visited);
                if (cur_level_id == 0) break;
                // Visited intentionally carries across layers — by the
                // monotone-distance argument (a rejected vid had dist >
                // the current top-k worst, and top-k only tightens),
                // skipping already-visited vids at lower layers cannot
                // miss a true NN.
            }

            const std::size_t k = std::min<std::size_t>(
                static_cast<std::size_t>(this->_topk),
                candidate_queue.get_result_size());
            if (k == 0) return knn_results_t{};
            return candidate_queue.extract_results(k);
        } else {
            // ---- Greedy upper levels + beam at L0 ----
            // `visited` is guaranteed clean — VisitedTablePool::acquire()
            // clears on every call, so each query gets a fresh table.
            auto [cursor_vid, cursor_dist] =
                candidate_sample_utils_t::sample_single_entry(
                    this->_vecs_data, this->_dist_func, hier_graph, query_vec);

            for (layer_id_t cur_level_id = top_level_id;
                 cur_level_id >= 1;
                 --cur_level_id)
            {
                std::tie(cursor_vid, cursor_dist) =
                    _single_layer_router.greedy_search(
                        query_vec, hier_graph, cur_level_id,
                        cursor_vid, cursor_dist, visited);
                if (cur_level_id == 1) break;
            }

            std_candidate_queue_t candidate_queue(
                static_cast<std::size_t>(_candidate_queue_size));
            candidate_queue.try_push(cursor_vid, cursor_dist);
            _single_layer_router.beam_search(
                query_vec, hier_graph, /*level_id=*/layer_id_t{0},
                candidate_queue, visited);

            const std::size_t k = std::min<std::size_t>(
                static_cast<std::size_t>(this->_topk),
                candidate_queue.get_result_size());
            if (k == 0) return knn_results_t{};
            return candidate_queue.extract_results(k);
        }
    }

    /**
     * @brief Parallel batch queries via TBB. Each worker acquires its
     *        own visited table from the pool. The @p UpperLevelBeamSearch
     *        template switch is forwarded to @c query verbatim.
     */
    template <bool UpperLevelBeamSearch = true, typename HierarchicalGraphT>
    auto batch_query(
        const query_vecs_t&       query_vecs,
        const HierarchicalGraphT& hier_graph
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
                        this->template query<UpperLevelBeamSearch>(q_vec, hier_graph);
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
    vertex_num_t                 _candidate_queue_size;
    mutable visited_table_pool_t _visited_table_pool;

};  // class HierarchicalGraphRouter

}   // namespace dynamic
}   // namespace cpu
}   // namespace artea
