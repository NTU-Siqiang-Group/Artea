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
 * @Description: Single-level router. Atom layer takes a NeighborRange;
 *               convenience layer queries a flat single-layer graph.
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <artea/common/logger.hpp>
#include <artea/cpu/router/detail/beam_loop.hpp>
#include <artea/cpu/router/detail/make_flat_range.hpp>
#include <artea/cpu/router/data_structures/neighbor_range_concept.hpp>
#include <artea/cpu/utils/parallel.hpp>

namespace artea {
namespace cpu {

/**
 * @brief Single-level beam-search primitive plus a convenience query
 *        layer over flat single-layer graphs.
 *
 * Atom layer: @c beam_search / @c greedy_search take a @c NeighborRange
 * adapter — graph-storage-agnostic, used by @c HierarchicalGraphRouter.
 *
 * Convenience layer: @c query / @c batch_query own a visited-table pool +
 * candidate-queue size + random sequence; flat adapter selected via
 * @c detail::make_flat_range based on the graph's @c is_compacted flag.
 *
 * @tparam RouterTraitsT The router traits type.
 */
template <typename RouterTraitsT, typename DistFuncT>
class SingleLayerRouter :
    public RouterTraitsT::template vector_router_t<DistFuncT, SingleLayerRouter<RouterTraitsT, DistFuncT>>
{

    using vertex_num_t          = typename RouterTraitsT::vertex_num_t;
    using vertex_id_t           = typename RouterTraitsT::vertex_id_t;
    using vec_id_t              = typename RouterTraitsT::vec_id_t;
    using vec_ele_t             = typename RouterTraitsT::vec_ele_t;
    using distance_t            = typename RouterTraitsT::distance_t;
    using dist_func_t           = DistFuncT;
    using vector_array_t        = typename RouterTraitsT::vector_array_t;
    using query_vecs_t          = typename RouterTraitsT::query_vecs_t;
    using nbr_arr_t             = typename RouterTraitsT::nbr_arr_t;
    using std_candidate_queue_t = typename RouterTraitsT::std_candidate_queue_t;
    using candidate_queue_t     = typename RouterTraitsT::candidate_queue_t;
    using visited_table_t       = typename RouterTraitsT::visited_table_t;
    using visited_table_pool_t  = typename RouterTraitsT::visited_table_pool_t;
    using random_seq_t          = typename RouterTraitsT::random_seq_t;
    using knn_results_t         = typename RouterTraitsT::knn_results_t;
    using base_class_t          =
        typename RouterTraitsT::template vector_router_t<DistFuncT, SingleLayerRouter<RouterTraitsT, DistFuncT>>;

public:
    /**
     * @brief Construct a SingleLayerRouter.
     *
     * @param topk                  Top-k for the convenience layer
     *                              (0 if only the atom API will be used).
     * @param candidate_queue_size  Beam width for the convenience layer
     *                              (must be >= @p topk if @p topk > 0).
     */
    SingleLayerRouter(
        const vector_array_t& base_vecs,
        const dist_func_t&    dist_func,
        const uint32_t        topk = 0,
        const vertex_num_t    candidate_queue_size = 0
    ) :
        base_class_t(base_vecs, dist_func, topk),
        _candidate_queue_size(candidate_queue_size),
        _visited_table_pool(base_vecs.get_num_vecs())
    {
        if (topk > 0 && _candidate_queue_size < topk) {
            ARTEA_ERROR(fmt::format(
                "candidate_queue_size ({}) must be >= topk ({})",
                _candidate_queue_size, topk));
        }
    }

    /** @brief Warm visited-table pool and the random sequence generator's
     *         per-thread MKL streams. Atom-only callers
     *         (@c HierarchicalGraphRouter) don't need to call this. */
    auto initialize() -> void {
        _visited_table_pool.warmup();
        _warmup_random_seq();
    }

    // ================================================================
    //   Atom layer (caller owns queue + visited lifecycle).
    // ================================================================

    /** @brief Greedy walk over @p nbrs_range starting from @p seed_vid;
     *         marks the seed in @p visited and returns the local optimum.
     *
     * `EnableFastL2 == true` switches the distance call to
     * @c dist_func.fast_euclidean(base_p, query, p_norm) and requires
     * @p base_norms to be the per-base ||p||^2 cache (typically from
     * the compact graph's @c get_base_norms()). EUCLIDEAN-only. */
    template <bool EnableFastL2 = false, NeighborRange NeighborRangeT,
              typename BaseNormsT = std::nullptr_t>
    __attribute__((always_inline))
    auto greedy_search(
        const vec_ele_t*       query_vec,
        const NeighborRangeT&  nbrs_range,
        vertex_id_t            seed_vid,
        distance_t             seed_dist,
        visited_table_t&       visited,
        const BaseNormsT&      base_norms = {}
    ) const -> std::pair<vertex_id_t, distance_t> {
        visited.set(seed_vid);
        return detail::greedy_loop_body<EnableFastL2, RouterTraitsT>(
            query_vec, nbrs_range, seed_vid, seed_dist,
            visited, this->_dist_func, this->_vecs_data, base_norms);
    }

    /**
     * @brief Beam search over @p nbrs_range, in-place on a pre-seeded
     *        @p candidate_queue. Marks current queue seeds in
     *        @p visited; does NOT clear @p visited (caller owns).
     *
     * @c reset_exploration() MUST run BEFORE the @c empty() check:
     * the unexplored heap drains across each call, so checking
     * @c empty() first would skip the level under shared-queue
     * hierarchical descent.
     *
     * `EnableFastL2 == true` switches the distance call to
     * @c dist_func.fast_euclidean(base_p, query, p_norm); see
     * @c greedy_search for the same opt-in.
     */
    template <bool EnableFastL2 = false, NeighborRange NeighborRangeT,
              typename BaseNormsT = std::nullptr_t>
    __attribute__((always_inline))
    auto beam_search(
        const vec_ele_t*       query_vec,
        const NeighborRangeT&  nbrs_range,
        std_candidate_queue_t& candidate_queue,
        visited_table_t&       visited,
        const BaseNormsT&      base_norms = {}
    ) const -> void {
        candidate_queue.reset_exploration();
        if (candidate_queue.empty()) return;

        for (const auto& seed : candidate_queue) {
            visited.set(seed.get_vid());
        }

        detail::beam_loop_body<EnableFastL2, RouterTraitsT>(
            query_vec, nbrs_range, candidate_queue,
            visited, this->_dist_func, this->_vecs_data, base_norms);
    }

    // ================================================================
    //   Convenience layer (flat single-layer graphs only).
    // ================================================================

    /**
     * @brief Query top-k. Picks a uniformly random vid that participates
     *        in @p single_layer_graph as the entry point — works for
     *        both dense and sparse upper-layer graphs via
     *        @c get_storage_vid.
     */
    template <typename SingleLayerGraphT>
    __attribute__((always_inline))
    auto query(
        const vec_ele_t*         query_vec,
        const SingleLayerGraphT& single_layer_graph
    ) const -> knn_results_t {
        auto& visited = _visited_table_pool.acquire();
        return _query_impl(query_vec, single_layer_graph,
                           _pick_random_entry(single_layer_graph), visited);
    }

    /** @brief Query top-k starting from a single explicit @p entry_point. */
    template <typename SingleLayerGraphT>
    __attribute__((always_inline))
    auto query(
        const vec_ele_t*         query_vec,
        const SingleLayerGraphT& single_layer_graph,
        const vertex_id_t        entry_point
    ) const -> knn_results_t {
        auto& visited = _visited_table_pool.acquire();
        return _query_impl(query_vec, single_layer_graph, entry_point, visited);
    }

    /**
     * @brief Query top-k with the candidate queue pre-seeded from a
     *        sorted, distance-bearing neighbor list. Skips the per-seed
     *        @c dist_func call. Dynamic graphs only — compact graphs
     *        store raw vids without distances.
     */
    template <typename SingleLayerGraphT>
    __attribute__((always_inline))
    auto query(
        const vec_ele_t*         query_vec,
        const SingleLayerGraphT& single_layer_graph,
        const nbr_arr_t&         seed_nbrs
    ) const -> knn_results_t {
        static_assert(!SingleLayerGraphT::is_compacted,
                      "warm-start query(seed_nbrs) requires distance-bearing "
                      "neighbor entries — only valid for dynamic graphs.");
        auto& visited = _visited_table_pool.acquire();
        return _query_impl(query_vec, single_layer_graph, seed_nbrs, visited);
    }

    /** @brief Parallel batch form of @c query(vec, graph) — each query
     *         picks its own random participating entry. */
    template <typename SingleLayerGraphT>
    auto batch_query(
        const query_vecs_t&      query_vecs,
        const SingleLayerGraphT& single_layer_graph
    ) const -> knn_results_t {
        const vertex_num_t num_queries = query_vecs.get_num_vecs();
        const uint32_t     k           = this->_topk;
        knn_results_t results(num_queries * k);

        tbb::parallel_for(
            tbb::blocked_range<vertex_num_t>(0, num_queries),
            [&](const tbb::blocked_range<vertex_num_t>& r) {
                for (vertex_num_t i = r.begin(); i != r.end(); ++i) {
                    auto& visited = _visited_table_pool.acquire();
                    auto topk_results = _query_impl(query_vecs.get(i), single_layer_graph,
                        _pick_random_entry(single_layer_graph), visited);
                    std::copy(topk_results.begin(), topk_results.end(), results.begin() + i * k);
                }
            }
        );
        return results;
    }

    /** @brief Parallel batch form of @c query(vec, graph, entry_point). */
    template <typename SingleLayerGraphT>
    auto batch_query(
        const query_vecs_t&      query_vecs,
        const SingleLayerGraphT& single_layer_graph,
        const vertex_id_t        entry_point
    ) const -> knn_results_t {
        const vertex_num_t num_queries = query_vecs.get_num_vecs();
        const uint32_t     k           = this->_topk;
        knn_results_t results(num_queries * k);

        tbb::parallel_for(
            tbb::blocked_range<vertex_num_t>(0, num_queries),
            [&](const tbb::blocked_range<vertex_num_t>& r) {
                for (vertex_num_t i = r.begin(); i != r.end(); ++i) {
                    auto& visited = _visited_table_pool.acquire();
                    auto topk_results = _query_impl(query_vecs.get(i), single_layer_graph,
                        entry_point, visited);
                    std::copy(topk_results.begin(), topk_results.end(), results.begin() + i * k);
                }
            }
        );
        return results;
    }

private:
    /** @brief Single-entry beam search on a freshly-acquired visited. */
    template <typename SingleLayerGraphT>
    __attribute__((always_inline))
    auto _query_impl(
        const vec_ele_t*         query_vec,
        const SingleLayerGraphT& single_layer_graph,
        const vertex_id_t        entry_point,
        visited_table_t&         visited
    ) const -> knn_results_t {
        const vertex_num_t queue_capacity = std::max(this->_topk, _candidate_queue_size);
        candidate_queue_t candidate_queue(queue_capacity);

        const distance_t entry_dist = this->_dist_func(
            query_vec, this->_vecs_data.get(entry_point));
        candidate_queue.try_push(entry_point, entry_dist);
        visited.set(entry_point);

        detail::beam_loop_body<false, RouterTraitsT>(
            query_vec,
            detail::make_flat_range(single_layer_graph),
            candidate_queue, visited, this->_dist_func, this->_vecs_data);

        return candidate_queue.extract_results(this->_topk);
    }

    /** @brief Warm-start beam search seeded from a sorted, distance-
     *         bearing neighbor list. Seeding loop walks the full
     *         @p seed_nbrs (uncapped). */
    template <typename SingleLayerGraphT>
    __attribute__((always_inline))
    auto _query_impl(
        const vec_ele_t*         query_vec,
        const SingleLayerGraphT& single_layer_graph,
        const nbr_arr_t&         seed_nbrs,
        visited_table_t&         visited
    ) const -> knn_results_t {
        const vertex_num_t queue_capacity = std::max(this->_topk, _candidate_queue_size);
        candidate_queue_t candidate_queue(queue_capacity);

        for (vertex_num_t i = 0;
             i < static_cast<vertex_num_t>(seed_nbrs.size());
             ++i)
        {
            const vertex_id_t seed_vid = seed_nbrs[i].get_vid();
            if (seed_vid == RouterTraitsT::invalid_vertex_id) break;
            if (visited.test_and_set(seed_vid)) continue;
            candidate_queue.try_push(seed_vid, seed_nbrs[i].get_distance());
        }

        detail::beam_loop_body<false, RouterTraitsT>(
            query_vec,
            detail::make_flat_range(single_layer_graph),
            candidate_queue, visited, this->_dist_func, this->_vecs_data);

        return candidate_queue.extract_results(this->_topk);
    }

    /** @brief Pick a uniformly random vid that participates in
     *         @p single_layer_graph. Works for both dense and sparse
     *         (upper-layer) graphs because @c get_storage_vid
     *         translates a local row index to a participating storage vid. */
    template <typename SingleLayerGraphT>
    __attribute__((always_inline))
    auto _pick_random_entry(const SingleLayerGraphT& single_layer_graph) const -> vertex_id_t {
        vec_id_t local_idx = 0;
        _random_seq.generate(&local_idx,
                             single_layer_graph.get_num_vertices(),
                             /*num=*/static_cast<vertex_num_t>(1));
        return single_layer_graph.get_storage_vid(static_cast<vertex_num_t>(local_idx));
    }

    /** @brief Trigger thread-local MKL stream creation on every TBB
     *         worker so the first real query doesn't pay the init cost. */
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

    vertex_num_t                 _candidate_queue_size;
    mutable visited_table_pool_t _visited_table_pool;
    mutable random_seq_t         _random_seq;

};  // class SingleLayerRouter

}   // namespace cpu
}   // namespace artea
