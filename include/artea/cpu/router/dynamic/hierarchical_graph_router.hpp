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
 * @Description: Multi-layer router over HierarchicalGraph (whose layers are
 *               InternalGraphs). Composes an InternalGraphRouter for the
 *               per-layer atom, then stacks a top-down descent (beam or
 *               greedy) on top. Also exposes the per-layer primitive as a
 *               build-time interface for stacked_rgraph::IndexFactory.
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <unordered_set>
#include <utility>
#include <vector>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <artea/common/logger.hpp>
#include <artea/cpu/utils/parallel.hpp>

namespace artea {
namespace cpu {
namespace dynamic {

/**
 * @brief Multi-layer proximity-graph router over a @c HierarchicalGraph.
 *
 * Composes an @c InternalGraphRouter internally so both routers share one
 * implementation of the per-layer beam-search atom. The hierarchical router
 * adds two things on top of that atom:
 *   - **Query path**: a top-down descent (@c beam_search / @c greedy_search)
 *     that seeds the top layer with uniformly random vertices, propagates
 *     the FULL per-layer candidate set down via @c get_inter_layer_link,
 *     and returns a @c knn_results_t from the bottom layer.
 *   - **Build path**: a public @c beam_search primitive that
 *     @c stacked_rgraph::IndexFactory calls from its insertion hot path to
 *     replace its hand-rolled per-layer beam search.
 *
 * The graph is NOT stored by reference. Every public method takes the
 * @c HierarchicalGraphT / @c InternalGraphT by template parameter at call
 * time.
 *
 * @tparam RouterTraitsT The router traits type.
 */
template <typename RouterTraitsT>
class HierarchicalGraphRouter :
    public RouterTraitsT::template vector_router_t<HierarchicalGraphRouter<RouterTraitsT>>
{

    using vertex_num_t            = typename RouterTraitsT::vertex_num_t;
    using vertex_id_t             = typename RouterTraitsT::vertex_id_t;
    using layer_num_t             = typename RouterTraitsT::layer_num_t;
    using layer_id_t              = typename RouterTraitsT::layer_id_t;
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
    using base_class_t            = typename RouterTraitsT::template vector_router_t<HierarchicalGraphRouter<RouterTraitsT>>;

    using inner_router_t          = InternalGraphRouter<RouterTraitsT>;

    static constexpr vertex_id_t invalid_vertex_id = RouterTraitsT::invalid_vertex_id;

public:
    /**
     * @brief Construct a HierarchicalGraphRouter.
     *
     * @param base_vecs             Base dataset (indexed by base_vid).
     * @param dist_func             Distance function functor.
     * @param topk                  Top-k returned by query / batch_query.
     * @param search_nn_qs          Beam width used at each layer during the
     *                              top-down descent. Also the number of
     *                              random seed vertices drawn from the top
     *                              layer.
     * @param candidate_queue_size  Candidate-queue capacity for the inner
     *                              single-layer router. Should be >= topk.
     */
    HierarchicalGraphRouter(
        const vector_array_t& base_vecs,
        const dist_func_t& dist_func,
        const uint32_t topk,
        const vertex_num_t search_nn_qs,
        const vertex_num_t candidate_queue_size = 16
    ) : base_class_t(base_vecs, dist_func, topk),
        _per_layer_router(base_vecs, dist_func, topk,
                   std::max(search_nn_qs, candidate_queue_size)),
        _search_nn_qs(search_nn_qs),
        _visited_table_pool(base_vecs.get_num_vecs())
    {
        if (search_nn_qs == 0) {
            ARTEA_ERROR("search_nn_qs must be >= 1");
        }
        if (candidate_queue_size < topk) {
            ARTEA_ERROR(fmt::format(
                "candidate_queue_size ({}) must be >= topk ({})",
                candidate_queue_size, topk
            ));
        }
    }

    /**
     * @brief Warm up the visited-table pool and the random-sequence
     *        generator for the query path. NOT required on the build path.
     */
    auto initialize() -> void {
        _visited_table_pool.warmup();
        _per_layer_router.initialize();
    }

    // ================================================================
    //   Build-time primitive (for stacked_rgraph::IndexFactory)
    // ================================================================

    /**
     * @brief Single-layer beam-search primitive. Trivially delegates to the
     *        composed @c InternalGraphRouter.
     *
     * See @c InternalGraphRouter::beam_search for semantics and
     * torn-read guards. This overload exists so the IndexFactory can hold
     * a single @c hierarchical_graph_router_t and call into the per-layer
     * atom without plumbing two router instances.
     */
    template <typename InternalGraphT>
    __attribute__((always_inline))
    auto beam_search(
        const vec_ele_t* query_vec,
        const InternalGraphT& layer,
        std_candidate_queue_t& candidate_queue,
        visited_table_t& visited
    ) const -> void {
        _per_layer_router.beam_search(query_vec, layer, candidate_queue, visited);
    }

    // ================================================================
    //   Query API
    // ================================================================

    /**
     * @brief Top-down beam search across every layer.
     *
     * 1. Randomly sample @c _search_nn_qs seed entries from the top layer.
     * 2. For h = h_max .. 1:
     *    - Run the per-layer atom with beam width @c _search_nn_qs. The
     *      atom loops until @c should_terminate().
     *    - If h > 1, translate EVERY candidate's layer_vid via
     *      @c get_inter_layer_link() into a fresh inbr_t for the next
     *      layer. Pass the entire result set down (no top-k slicing).
     * 3. At h == 1, trim the final sorted result set to @c this->_topk.
     */
    template <typename HierarchicalGraphT>
    auto beam_search(
        const vec_ele_t* query_vec,
        const HierarchicalGraphT& hg
    ) const -> knn_results_t {
        const layer_num_t h_max = hg.get_num_layers();
        if (h_max == 0) return _empty_results();

        auto& visited = _visited_table_pool.acquire();
        const std::size_t L = static_cast<std::size_t>(_search_nn_qs);

        // --- Top-layer random seeding ---
        std_candidate_queue_t candidate_queue(L);
        {
            const auto& top_layer = hg.get_layer_graph(h_max - 1);
            _per_layer_router.sample_entries(top_layer, query_vec, candidate_queue);
        }

        if (candidate_queue.empty()) {
            visited.clear();
            return _empty_results();
        }

        // --- Top-down descent ---
        for (layer_num_t h = h_max; h >= 1; --h) {
            const layer_id_t layer_idx = h - 1;
            const auto& layer = hg.get_layer_graph(layer_idx);

            _per_layer_router.beam_search(query_vec, layer, candidate_queue, visited);

            if (candidate_queue.get_result_size() == 0) break;

            if (h > 1) {
                // Build next-layer queue via inter-layer link remapping.
                std_candidate_queue_t next_candidate_queue(L);
                for (const auto& c : candidate_queue) {
                    const vertex_id_t lower_lv =
                        layer.get_inter_layer_link(c.get_layer_vid());
                    next_candidate_queue.try_push(c.get_base_vid(), lower_lv, c.get_distance());
                }
                candidate_queue = std::move(next_candidate_queue);
            }
        }

        visited.clear();
        const std::size_t topk = std::min(
            static_cast<std::size_t>(this->_topk), candidate_queue.get_result_size());
        if (topk == 0) return _empty_results();
        return candidate_queue.extract_results(topk);
    }

    /**
     * @brief Top-down greedy descent across every layer.
     *
     * 1. Sample a single random entry from the top layer.
     * 2. For h = h_max .. 1:
     *    - Greedy hill-climb on this layer with one best vertex. Advance
     *      to the strictly-closest neighbor until no improvement, then
     *      descend via get_inter_layer_link() to the next layer.
     * 3. At h == 1, run one final beam search seeded from the greedy
     *    endpoint to collect a proper top-k around it.
     */
    template <typename HierarchicalGraphT>
    auto greedy_search(
        const vec_ele_t* query_vec,
        const HierarchicalGraphT& hg
    ) const -> knn_results_t {
        const layer_num_t h_max = hg.get_num_layers();
        if (h_max == 0) return _empty_results();

        auto& visited = _visited_table_pool.acquire();

        // --- Top-layer single random entry ---
        candidate_entry_t current;
        {
            const auto& top_layer = hg.get_layer_graph(h_max - 1);
            const vertex_num_t top_size = top_layer.get_num_vertices();
            if (top_size == 0) {
                visited.clear();
                return _empty_results();
            }

            const vertex_id_t lv = _draw_random_vid(top_size);
            const vertex_id_t bv = top_layer.get_base_vid(lv);
            const distance_t dist =
                this->_dist_func(query_vec, this->_vecs_data.get(bv));
            current = candidate_entry_t(bv, lv, dist);
        }

        // --- Top-down greedy descent ---
        for (layer_num_t h = h_max; h >= 1; --h) {
            const layer_id_t layer_idx = h - 1;
            const auto& layer = hg.get_layer_graph(layer_idx);

            current = _per_layer_router.greedy_search(
                query_vec, layer, current, visited);

            if (h > 1) {
                const vertex_id_t lower_lv =
                    layer.get_inter_layer_link(current.get_layer_vid());
                current = candidate_entry_t(
                    current.get_base_vid(), lower_lv, current.get_distance());
            }
        }

        // --- Final beam search around the greedy endpoint on the bottom layer ---
        const auto& bottom = hg.get_layer_graph(0);
        std_candidate_queue_t candidate_queue(static_cast<std::size_t>(_search_nn_qs));
        candidate_queue.try_push(current.get_base_vid(), current.get_layer_vid(),
                     current.get_distance());
        _per_layer_router.beam_search(query_vec, bottom, candidate_queue, visited);

        visited.clear();
        const std::size_t topk = std::min(
            static_cast<std::size_t>(this->_topk), candidate_queue.get_result_size());
        if (topk == 0) return _empty_results();
        return candidate_queue.extract_results(topk);
    }

    /**
     * @brief Query the top-k nearest vertices. Thin wrapper over
     *        @c beam_search().
     */
    template <typename HierarchicalGraphT>
    __attribute__((always_inline))
    auto query(
        const vec_ele_t* query_vec,
        const HierarchicalGraphT& hg
    ) const -> knn_results_t {
        return beam_search(query_vec, hg);
    }

    /**
     * @brief Batch queries. Parallelized over queries with TBB.
     */
    template <typename HierarchicalGraphT>
    auto batch_query(
        const query_vecs_t& query_vecs,
        const HierarchicalGraphT& hg
    ) const -> knn_results_t {
        const vertex_num_t num_queries = query_vecs.get_num_vecs();
        const uint32_t K = this->_topk;

        knn_results_t results(num_queries * K);

        tbb::parallel_for(
            tbb::blocked_range<vertex_num_t>(0, num_queries),
            [&](const tbb::blocked_range<vertex_num_t>& r) {
                for (vertex_num_t i = r.begin(); i != r.end(); ++i) {
                    const vec_ele_t* q_vec = query_vecs.get(i);
                    auto topk_results = this->query(q_vec, hg);
                    std::copy(topk_results.begin(), topk_results.end(),
                              results.begin() + i * K);
                }
            }
        );

        return results;
    }

    /**
     * @brief Public delegate so external callers (e.g.
     *        @c stacked_rgraph::IndexFactory) can sample random entries
     *        from a layer through the hierarchical router without
     *        plumbing two router instances. Forwards to the composed
     *        @c InternalGraphRouter::sample_entries.
     */
    template <typename InternalGraphT>
    __attribute__((always_inline))
    auto sample_entries(
        const InternalGraphT& layer,
        const vec_ele_t* query_vec,
        std_candidate_queue_t& candidate_queue
    ) const -> void {
        _per_layer_router.sample_entries(layer, query_vec, candidate_queue);
    }

private:
    // ---- Helpers ----

    auto _empty_results() const -> knn_results_t {
        const std::size_t K = static_cast<std::size_t>(this->_topk);
        return knn_results_t(K, candidate_entry_t::make_invalid_entry());
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

    /** @brief Composed per-layer router. Owns the beam-search atom. */
    mutable inner_router_t _per_layer_router;

    /** @brief Beam width used at each layer during the top-down descent. */
    vertex_num_t _search_nn_qs = 0;

    /** @brief Pool of thread-local visited bitmaps for parallel queries. */
    mutable visited_table_pool_t _visited_table_pool;

};  // class HierarchicalGraphRouter

}   // namespace dynamic
}   // namespace cpu
}   // namespace artea
