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
 *   - **Build-time passthroughs** — @c beam_search(query, hg, level_id,
 *     queue, visited) and @c sample_entries(hg, level_id, query, queue)
 *     that simply delegate to the composed @c SingleLayerRouter. Used by
 *     @c stacked_rgraph::IndexFactory on the insertion hot path.
 *   - **Query path** — @c beam_search(query, hg) runs the full top-down
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
    //   Build-time passthroughs
    // ================================================================

    template <typename HierarchicalGraphT>
    __attribute__((always_inline))
    auto beam_search(
        const vec_ele_t*          query_vec,
        const HierarchicalGraphT& hg,
        const layer_id_t          level_id,
        std_candidate_queue_t&    candidate_queue,
        visited_table_t&          visited
    ) const -> void {
        _single_layer_router.beam_search(
            query_vec, hg, level_id, candidate_queue, visited);
    }

    template <typename HierarchicalGraphT>
    __attribute__((always_inline))
    auto sample_entries(
        const HierarchicalGraphT& hg,
        const layer_id_t          level_id,
        const vec_ele_t*          query_vec,
        std_candidate_queue_t&    candidate_queue
    ) const -> void {
        _single_layer_router.sample_entries(
            hg, level_id, query_vec, candidate_queue);
    }

    // ================================================================
    //   Query path
    // ================================================================

    /**
     * @brief Top-down beam search across every level.
     *
     *   1. Locate the top occupied level.
     *   2. Seed a fresh candidate queue with random vertices at that
     *      level via @c sample_entries.
     *   3. For @c cur_level = top .. 0, run
     *      @c _single_layer_router.beam_search on the same queue.
     *      No inter-level translation is needed because every vid is
     *      valid at every level it participates in.
     *   4. Trim the result set to @c topk.
     */
    template <typename HierarchicalGraphT>
    auto beam_search(
        const vec_ele_t*          query_vec,
        const HierarchicalGraphT& hg
    ) const -> knn_results_t {
        const layer_id_t top_level_id = hg.top_occupied_highest_level_id();
        if (top_level_id == HierarchicalGraphT::unassigned_highest_level_id) {
            return knn_results_t{};
        }

        auto& visited = _visited_table_pool.acquire();
        std_candidate_queue_t candidate_queue(
            static_cast<std::size_t>(_candidate_queue_size));

        _single_layer_router.sample_entries(
            hg, top_level_id, query_vec, candidate_queue);

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
    }

    /** @brief Thin alias for @c beam_search(query_vec, hg). */
    template <typename HierarchicalGraphT>
    __attribute__((always_inline))
    auto query(
        const vec_ele_t*          query_vec,
        const HierarchicalGraphT& hg
    ) const -> knn_results_t {
        return beam_search(query_vec, hg);
    }

    /**
     * @brief Parallel batch queries via TBB. Each worker acquires its
     *        own visited table from the pool.
     */
    template <typename HierarchicalGraphT>
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
                    auto topk_results = this->query(q_vec, hg);
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

}   // namespace dynamic
}   // namespace cpu
}   // namespace artea
