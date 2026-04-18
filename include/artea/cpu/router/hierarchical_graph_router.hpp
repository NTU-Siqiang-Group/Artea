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
 * @FilePath: /Artea/include/artea/cpu/router/hierarchical_graph_router.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Unified multi-level router. Graph-storage-agnostic at the
 *               class level: the graph type enters as a per-method
 *               template parameter, and the right NeighborRange adapter
 *               is selected via @c detail::make_layer_range based on the
 *               graph's @c is_compacted flag. Replaces the per-mode
 *               compact::HierarchicalGraphRouter and
 *               dynamic::HierarchicalGraphRouter.
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <artea/common/logger.hpp>
#include <artea/cpu/router/detail/make_layer_range.hpp>
#include <artea/cpu/router/single_layer_router.hpp>
#include <artea/cpu/utils/parallel.hpp>

namespace artea {
namespace cpu {

/**
 * @brief Multi-level proximity-graph router. Composes a
 *        @c SingleLayerRouter for the per-level atom; cross-level
 *        composition is a straight top-down descent (no inter-layer
 *        candidate translation because every vertex has one global vid
 *        that is valid at every level it participates in).
 *
 * Provides both:
 *   - **Build-time passthrough** — @c beam_search(query, graph, level_id,
 *     queue, visited) used by @c stacked_rgraph::IndexFactory on the
 *     insertion hot path.
 *   - **Query path** — @c query / @c batch_query / @c query_l0_only.
 *
 * @tparam RouterTraitsT The router traits type.
 */
template <typename RouterTraitsT>
class HierarchicalGraphRouter :
    public RouterTraitsT::template vector_router_t<HierarchicalGraphRouter<RouterTraitsT>>
{
    using vertex_num_t             = typename RouterTraitsT::vertex_num_t;
    using vertex_id_t              = typename RouterTraitsT::vertex_id_t;
    using layer_id_t               = typename RouterTraitsT::layer_id_t;
    using vec_ele_t                = typename RouterTraitsT::vec_ele_t;
    using distance_t               = typename RouterTraitsT::distance_t;
    using dist_func_t              = typename RouterTraitsT::dist_func_t;
    using vector_array_t           = typename RouterTraitsT::vector_array_t;
    using query_vecs_t             = typename RouterTraitsT::query_vecs_t;
    using candidate_entry_t        = typename RouterTraitsT::candidate_entry_t;
    using std_candidate_queue_t    = typename RouterTraitsT::std_candidate_queue_t;
    using visited_table_t          = typename RouterTraitsT::visited_table_t;
    using visited_table_pool_t     = typename RouterTraitsT::visited_table_pool_t;
    using knn_results_t            = typename RouterTraitsT::knn_results_t;
    using random_seq_t             = typename RouterTraitsT::random_seq_t;
    using candidate_sample_utils_t = typename RouterTraitsT::candidate_sample_utils_t;
    using base_class_t             =
        typename RouterTraitsT::template vector_router_t<HierarchicalGraphRouter<RouterTraitsT>>;

    using single_layer_router_t    = SingleLayerRouter<RouterTraitsT>;

    static constexpr vertex_id_t invalid_vertex_id = RouterTraitsT::invalid_vertex_id;

public:
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

    /** @brief Warm up the visited-table pool and (for query paths that
     *         use random init) the random sequence generator. Cheap on
     *         dynamic-only callers — they just never touch _random_seq. */
    auto initialize() -> void {
        _visited_table_pool.warmup();
        _warmup_random_seq();
    }

    // ================================================================
    //   Build-time passthrough
    // ================================================================

    /** @brief Single-level beam search at @p level_id. Constructs the
     *         right NeighborRange adapter for @p hier_graph and forwards
     *         to the per-level atom. Used by
     *         @c stacked_rgraph::IndexFactory on the insertion path. */
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
            query_vec,
            detail::make_layer_range(hier_graph, level_id),
            candidate_queue, visited);
    }

    // ================================================================
    //   Query path
    // ================================================================

    /**
     * @brief Top-down hierarchical search.
     *
     * @tparam RandomSeeding
     *   - @c false (default): seed from the graph's precomputed
     *     @c entry_point_vid (compact graphs). Dynamic graphs have no
     *     precomputed entry point, so this falls back to
     *     @c sample_single_entry automatically.
     *   - @c true: always seed via @c sample_single_entry from the apex
     *     bucket.
     *
     * @tparam UpperLevelBeamSearch
     *   - @c false (default): HNSW-style cheap @c greedy_search on
     *     levels @c top..1 with a single cursor, then @c beam_search at
     *     L0 for the actual top-K result.
     *   - @c true: shared-queue @c beam_search at every level from
     *     @c top..0.
     */
    template <bool RandomSeeding = false, bool UpperLevelBeamSearch = false,
              typename HierarchicalGraphT>
    auto query(
        const vec_ele_t*          query_vec,
        const HierarchicalGraphT& hier_graph
    ) const -> knn_results_t {
        const layer_id_t top_level_id = hier_graph.top_occupied_level_id();
        if (top_level_id == HierarchicalGraphT::unassigned_highest_level_id) {
            return knn_results_t{};
        }

        auto& visited = _visited_table_pool.acquire();

        // ---- Pick the entry point ----
        vertex_id_t entry_vid;
        distance_t  entry_dist;
        if constexpr (!RandomSeeding && HierarchicalGraphT::is_compacted) {
            entry_vid  = hier_graph.entry_point_vid();
            entry_dist = this->_dist_func(query_vec, this->_vecs_data.get(entry_vid));
        } else {
            // Either RandomSeeding=true, or the graph is dynamic (no
            // precomputed entry point) — sample one fresh from the apex.
            std::tie(entry_vid, entry_dist) =
                candidate_sample_utils_t::sample_single_entry(
                    this->_vecs_data, this->_dist_func, hier_graph, query_vec);
        }

        // ---- Search phase ----
        if constexpr (UpperLevelBeamSearch) {
            std_candidate_queue_t candidate_queue(
                static_cast<std::size_t>(_candidate_queue_size));
            candidate_queue.try_push(entry_vid, entry_dist);

            for (layer_id_t cur_level_id = top_level_id; ; --cur_level_id) {
                _single_layer_router.beam_search(
                    query_vec,
                    detail::make_layer_range(hier_graph, cur_level_id),
                    candidate_queue, visited);
                if (cur_level_id == 0) break;
                // Reset visited between layers — each level walks a
                // different neighborhood graph; cheap with VersionTagTable.
                visited.clear();
            }

            const std::size_t k = std::min<std::size_t>(
                static_cast<std::size_t>(this->_topk),
                candidate_queue.get_result_size());
            if (k == 0) return knn_results_t{};
            return candidate_queue.extract_results(k);
        } else {
            vertex_id_t cursor_vid  = entry_vid;
            distance_t  cursor_dist = entry_dist;
            // Clear after every greedy level (L1 included) so the L0
            // beam below always starts with a clean visited. The L0
            // clear is mandatory: greedy tracks a single best cursor,
            // while L0 beam targets top-K; some L1-rejected vids could
            // legitimately enter L0 top-K. O(1) per clear with
            // VersionTagTable.
            for (layer_id_t cur_level_id = top_level_id;
                 cur_level_id >= 1;
                 --cur_level_id)
            {
                std::tie(cursor_vid, cursor_dist) =
                    _single_layer_router.greedy_search(
                        query_vec,
                        detail::make_layer_range(hier_graph, cur_level_id),
                        cursor_vid, cursor_dist, visited);
                visited.clear();
            }

            std_candidate_queue_t candidate_queue(
                static_cast<std::size_t>(_candidate_queue_size));
            candidate_queue.try_push(cursor_vid, cursor_dist);
            _single_layer_router.beam_search(
                query_vec,
                detail::make_layer_range(hier_graph, layer_id_t{0}),
                candidate_queue, visited);

            const std::size_t k = std::min<std::size_t>(
                static_cast<std::size_t>(this->_topk),
                candidate_queue.get_result_size());
            if (k == 0) return knn_results_t{};
            return candidate_queue.extract_results(k);
        }
    }

    /**
     * @brief L0-only baseline: skip the hierarchy entirely and run a
     *        single beam search on the base layer. Compact-only —
     *        seeding requires either the precomputed @c entry_point_vid
     *        or @c random_initialize on the candidate queue, neither of
     *        which makes sense for the dynamic mid-build state.
     */
    template <bool RandomSeeding = false, typename HierarchicalGraphT>
    auto query_l0_only(
        const vec_ele_t*          query_vec,
        const HierarchicalGraphT& hier_graph
    ) const -> knn_results_t {
        static_assert(HierarchicalGraphT::is_compacted,
                      "query_l0_only is only defined for compact graphs.");

        auto& visited = _visited_table_pool.acquire();

        std_candidate_queue_t candidate_queue(
            static_cast<std::size_t>(_candidate_queue_size));

        if constexpr (RandomSeeding) {
            candidate_queue.random_initialize(
                _random_seq, this->_dist_func, query_vec,
                this->_vecs_data, visited);
        } else {
            const vertex_id_t entry_vid  = hier_graph.entry_point_vid();
            const distance_t  entry_dist = this->_dist_func(
                query_vec, this->_vecs_data.get(entry_vid));
            candidate_queue.try_push(entry_vid, entry_dist);
        }

        _single_layer_router.beam_search(
            query_vec,
            detail::make_layer_range(hier_graph, layer_id_t{0}),
            candidate_queue, visited);

        const std::size_t k = std::min<std::size_t>(
            static_cast<std::size_t>(this->_topk),
            candidate_queue.get_result_size());
        if (k == 0) return knn_results_t{};
        return candidate_queue.extract_results(k);
    }

    /**
     * @brief Parallel batch queries via TBB. Each worker acquires its
     *        own visited table from the pool. The template switches are
     *        forwarded to @c query verbatim.
     */
    template <bool RandomSeeding = false, bool UpperLevelBeamSearch = false,
              typename HierarchicalGraphT>
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
                        this->template query<RandomSeeding, UpperLevelBeamSearch>(
                            q_vec, hier_graph);
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

    /** @brief Parallel batch form of @c query_l0_only. Compact-only. */
    template <bool RandomSeeding = false, typename HierarchicalGraphT>
    auto batch_query_l0_only(
        const query_vecs_t&       query_vecs,
        const HierarchicalGraphT& hier_graph
    ) const -> knn_results_t {
        static_assert(HierarchicalGraphT::is_compacted,
                      "batch_query_l0_only is only defined for compact graphs.");

        const vertex_num_t num_queries = query_vecs.get_num_vecs();
        const uint32_t     k           = this->_topk;

        knn_results_t results(num_queries * k);

        tbb::parallel_for(
            tbb::blocked_range<vertex_num_t>(0, num_queries),
            [&](const tbb::blocked_range<vertex_num_t>& r) {
                for (vertex_num_t i = r.begin(); i != r.end(); ++i) {
                    const vec_ele_t* q_vec = query_vecs.get(i);
                    auto topk_results =
                        this->template query_l0_only<RandomSeeding>(q_vec, hier_graph);
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
    /** @brief Trigger thread-local MKL stream creation for the random
     *         sequence generator on every TBB worker, so the first real
     *         query doesn't pay the init cost. */
    __attribute__((always_inline))
    auto _warmup_random_seq() -> void {
        const int num_threads = tbb_max_num_threads();
        tbb::parallel_for(
            tbb::blocked_range<int>(0, num_threads, 1),
            [&](const tbb::blocked_range<int>&) {
                std::vector<vertex_id_t> dummy(1);
                _random_seq.generate(dummy, /*upper_bound=*/1, /*num=*/1);
            }
        );
    }

    single_layer_router_t        _single_layer_router;
    vertex_num_t                 _candidate_queue_size;
    mutable visited_table_pool_t _visited_table_pool;
    mutable random_seq_t         _random_seq;

};  // class HierarchicalGraphRouter

}   // namespace cpu
}   // namespace artea
