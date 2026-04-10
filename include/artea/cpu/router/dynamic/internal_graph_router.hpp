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
 *               Uses a lnbr-backed StdCandidateQueue so each candidate
 *               carries both @c base_vid (for coordinate lookup / inter-layer
 *               descent) and @c layer_vid (for the visited table and CSR
 *               traversal) without a side-channel hash map.
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <unordered_set>
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
 * This router is the counterpart of @c DescentGraphRouter<...::compact_mode>
 * for the dynamic-insertion index family. It carries both @c base_vid and
 * @c layer_vid through the beam-search candidate queue (via
 * @c LnbrCandidateEntry), so callers — including
 * @c stacked_rgraph::IndexFactory on its build-time path — can recover both
 * identifiers directly from extracted results without a side-channel
 * @c std::unordered_map<layer_vid, base_vid>.
 *
 * The graph is NOT stored by reference. Every public method takes the
 * @c InternalGraphT to operate on as a template parameter at call time,
 * matching the pattern used by @c DescentGraphRouter<...::dynamic_mode>.
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
    using lnbr_t                  = typename RouterTraitsT::lnbr_t;
    using lnbr_candidate_entry_t  = typename RouterTraitsT::lnbr_candidate_entry_t;
    using dnbr_candidate_entry_t  = typename RouterTraitsT::dnbr_candidate_entry_t;
    using std_lnbr_candidate_queue_t =
        typename RouterTraitsT::std_lnbr_candidate_queue_t;
    using visited_table_t         = typename RouterTraitsT::visited_table_t;
    using visited_table_pool_t    = typename RouterTraitsT::visited_table_pool_t;
    using knn_results_t           = typename RouterTraitsT::knn_results_t;
    using random_seq_t            = typename RouterTraitsT::random_seq_t;
    using base_class_t            = typename RouterTraitsT::template vector_router_t<InternalGraphRouter<RouterTraitsT>>;

    static constexpr vertex_id_t invalid_vertex_id = RouterTraitsT::invalid_vertex_id;

public:
    /**
     * @brief Construct an InternalGraphRouter.
     *
     * @param base_vecs             Base dataset. Coordinates are fetched via
     *                              @c base_vecs.get(base_vid) inside
     *                              @c beam_search_layer and its callers.
     * @param dist_func             Distance function functor.
     * @param topk                  Number of nearest neighbors returned by
     *                              @c query() / @c batch_query() / @c beam_search().
     * @param candidate_queue_size  Beam width used by the high-level
     *                              @c beam_search() / @c query() entry
     *                              points. The low-level
     *                              @c beam_search_layer() takes its width
     *                              as an explicit argument.
     */
    InternalGraphRouter(
        const vector_array_t& base_vecs,
        const dist_func_t& dist_func,
        const uint32_t topk,
        const vertex_num_t candidate_queue_size = 16
    ) : base_class_t(base_vecs, dist_func, topk),
        _candidate_queue_size(candidate_queue_size),
        _visited_table_pool(base_vecs.get_num_vecs()),
        _random_seq()
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
     * primitive (@c beam_search_layer) does NOT require this — it takes a
     * caller-owned visited table and never samples random entries.
     */
    auto initialize() -> void {
        _visited_table_pool.warmup();
        _warmup_random_seq();
    }

    // ================================================================
    //   Build-time / atomic primitive
    // ================================================================

    /**
     * @brief Beam search on a single InternalGraph layer starting from a
     *        caller-provided entry set. Returns up to @p queue_width sorted
     *        @c lnbr_candidate_entry_t results.
     *
     * This is the primitive that @c stacked_rgraph::IndexFactory calls on
     * its insertion hot path. The caller owns the @c visited table (which
     * is cleared on entry) and the @p entries span (both @c base_vid and
     * @c layer_vid per entry are consumed).
     *
     * Torn-read guards (all three preserved from the original
     * stacked_rgraph implementation):
     *   - Entry @c layer_vid is clamped against @c layer.get_num_vertices()
     *     to ignore stale entries from a concurrent extension.
     *   - Raw neighbor count is clamped against @c layer.max_nbr_size() to
     *     tolerate partial header writes.
     *   - Neighbor @c layer_vid is clamped against the layer size to ignore
     *     torn CSR slots.
     *
     * Reads are lock-free: the per-vertex spinlock is NOT acquired. Stale
     * reads of newly-inserted neighbors are tolerated (per the paper's
     * optimistic-concurrency design).
     *
     * @tparam InternalGraphT The internal-graph type (pulled in at call
     *         time to keep header dependencies minimal).
     * @param query_vec   Pointer to the query vector data.
     * @param layer       The InternalGraph layer to search.
     * @param entries     Seed entry set (base_vid + layer_vid pairs).
     * @param queue_width Beam width (queue capacity).
     * @param visited     Caller-owned visited table. Cleared on entry;
     *                    may contain marks from prior calls.
     * @return Sorted @c std::vector<lnbr_candidate_entry_t> (ascending by
     *         distance). Never larger than @p queue_width.
     */
    template <typename InternalGraphT>
    auto beam_search_layer(
        const vec_ele_t* query_vec,
        const InternalGraphT& layer,
        std::span<const lnbr_t> entries,
        const vertex_num_t queue_width,
        visited_table_t& visited
    ) const -> std::vector<lnbr_candidate_entry_t> {
        const vertex_num_t layer_size = layer.get_num_vertices();
        if (layer_size == 0) return {};

        const vertex_num_t max_nbr = layer.max_nbr_size();
        const std::size_t  L = static_cast<std::size_t>(queue_width);

        visited.clear();
        std_lnbr_candidate_queue_t cq(L);

        // Seed with every (deduped, in-bounds) entry point.
        for (const lnbr_t& e : entries) {
            if (e.base_vid == invalid_vertex_id) continue;
            const vertex_id_t lv = e.layer_vid;
            if (lv >= layer_size) continue;  // stale entry, skip
            if (visited.test_and_set(lv)) continue;
            const distance_t d =
                this->_dist_func(query_vec, this->_vecs_data.get(e.base_vid));
            cq.try_push(e.base_vid, lv, d);
        }

        if (cq.empty()) return {};

        // Standard beam-search loop.
        while (!cq.empty()) {
            if (cq.should_terminate()) break;
            const lnbr_candidate_entry_t current = cq.pop_best_unexplored_entry();
            if (current.is_invalid()) break;

            const vertex_id_t cur_lv = current.get_layer_vid();

            const auto block = layer.fetch_nbrs(cur_lv);
            const uint64_t raw_count = layer.num_valid_nbrs(cur_lv);
            const uint64_t count = std::min<uint64_t>(raw_count, max_nbr);
            for (uint64_t i = 0; i < count; ++i) {
                const lnbr_t nbr = block[1 + i];
                if (nbr.base_vid == invalid_vertex_id) continue;
                if (nbr.layer_vid >= layer_size) continue;  // torn read
                if (visited.test_and_set(nbr.layer_vid)) continue;
                const distance_t d =
                    this->_dist_func(query_vec, this->_vecs_data.get(nbr.base_vid));
                cq.try_push(nbr.base_vid, nbr.layer_vid, d);
            }
        }

        // Extract sorted top-k results (clamped to the number actually held).
        const std::size_t k = std::min(L, cq.get_result_size());
        if (k == 0) return {};
        return cq.extract_results(k);
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

        std::vector<lnbr_t> entry_buf;
        sample_random_entries(layer, _candidate_queue_size, entry_buf);

        auto cands = beam_search_layer(
            query_vec, layer,
            std::span<const lnbr_t>(entry_buf),
            _candidate_queue_size, visited);

        visited.clear();
        return _to_knn_results(cands);
    }

    /**
     * @brief Greedy (best-first) hill-climb search on a single layer.
     *
     * Starts from a single random vertex, then repeatedly advances to the
     * strictly-closest neighbor of the current best until no neighbor can
     * improve the current distance. Returns the final
     * @c lnbr_candidate_entry_t (base_vid + layer_vid + distance).
     *
     * This is the per-layer primitive that a hierarchical greedy descent
     * composes across layers.
     */
    template <typename InternalGraphT>
    auto greedy_search(
        const vec_ele_t* query_vec,
        const InternalGraphT& layer
    ) const -> lnbr_candidate_entry_t {
        const vertex_num_t layer_size = layer.get_num_vertices();
        if (layer_size == 0) return lnbr_candidate_entry_t::make_invalid_entry();

        // Draw a single random layer_vid directly in [0, layer_size).
        std::vector<vertex_id_t> scratch(1);
        _random_seq.generate(scratch, /*upper_bound=*/layer_size, /*num=*/1);
        const vertex_id_t start_lv = scratch[0];
        const vertex_id_t start_bv = layer.get_base_vid(start_lv);
        const distance_t  start_d  =
            this->_dist_func(query_vec, this->_vecs_data.get(start_bv));

        return _greedy_from(query_vec, layer,
                            lnbr_candidate_entry_t(start_bv, start_lv, start_d));
    }

    /**
     * @brief Internal helper: greedy hill-climb starting from a given
     *        entry point. Exposed so the hierarchical router can call it
     *        directly without another random draw.
     */
    template <typename InternalGraphT>
    auto greedy_from_entry(
        const vec_ele_t* query_vec,
        const InternalGraphT& layer,
        const lnbr_candidate_entry_t& start
    ) const -> lnbr_candidate_entry_t {
        return _greedy_from(query_vec, layer, start);
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
                        results[i * K + j] = dnbr_candidate_entry_t::make_invalid_entry();
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
     * @brief Sample up to @p k distinct random entries from @p layer and
     *        write them into @p out as @c lnbr_t(base_vid, layer_vid) pairs.
     *
     * Uses @c random_seq_t to bulk-generate uniform integers in
     * @c [0, layer_size) directly. Duplicates are absorbed via a small
     * @c unordered_set; we over-generate by 2x to keep the rejection
     * probability bounded. If @p layer has fewer than @p k vertices,
     * every vertex is taken.
     *
     * Thread-safe: the underlying @c random_seq_t uses TBB
     * thread-local MKL streams, so multiple threads may invoke this
     * concurrently on the same router instance.
     */
    template <typename InternalGraphT>
    auto sample_random_entries(
        const InternalGraphT& layer,
        const vertex_num_t k,
        std::vector<lnbr_t>& out
    ) const -> void {
        const vertex_num_t layer_size = layer.get_num_vertices();
        out.clear();
        if (layer_size == 0) return;

        if (layer_size <= k) {
            out.reserve(layer_size);
            for (vertex_num_t lv = 0; lv < layer_size; ++lv) {
                out.emplace_back(layer.get_base_vid(lv), lv);
            }
            return;
        }

        // Over-generate a bit to absorb duplicate draws.
        const std::size_t raw_count = static_cast<std::size_t>(k) * 2;
        std::vector<vertex_id_t> buf(raw_count);
        _random_seq.generate(buf, /*upper_bound=*/layer_size,
                             /*num=*/static_cast<vertex_num_t>(raw_count));

        std::unordered_set<vertex_id_t> picked;
        picked.reserve(static_cast<std::size_t>(k));
        out.reserve(static_cast<std::size_t>(k));

        for (std::size_t i = 0; i < raw_count && picked.size() < static_cast<std::size_t>(k); ++i) {
            const vertex_id_t lv = buf[i];
            if (picked.insert(lv).second) {
                out.emplace_back(layer.get_base_vid(lv), lv);
            }
        }

        // Fallback: if over-generation still didn't yield k entries (rare),
        // top up with a deterministic linear probe.
        for (vertex_id_t lv = 0; out.size() < static_cast<std::size_t>(k) && lv < layer_size; ++lv) {
            if (picked.insert(lv).second) {
                out.emplace_back(layer.get_base_vid(lv), lv);
            }
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
        lnbr_candidate_entry_t current
    ) const -> lnbr_candidate_entry_t {
        const vertex_num_t layer_size = layer.get_num_vertices();
        const vertex_num_t max_nbr = layer.max_nbr_size();

        while (true) {
            const vertex_id_t cur_lv = current.get_layer_vid();
            if (cur_lv >= layer_size) break;  // stale/torn

            const auto block = layer.fetch_nbrs(cur_lv);
            const uint64_t raw_count = layer.num_valid_nbrs(cur_lv);
            const uint64_t count = std::min<uint64_t>(raw_count, max_nbr);

            lnbr_candidate_entry_t best = current;
            for (uint64_t i = 0; i < count; ++i) {
                const lnbr_t nbr = block[1 + i];
                if (nbr.base_vid == invalid_vertex_id) continue;
                if (nbr.layer_vid >= layer_size) continue;  // torn read
                const distance_t d =
                    this->_dist_func(query_vec, this->_vecs_data.get(nbr.base_vid));
                if (d < best.get_distance()) {
                    best = lnbr_candidate_entry_t(nbr.base_vid, nbr.layer_vid, d);
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
     * @brief Materialize a @c knn_results_t (= @c std::vector<dnbr_candidate_entry_t>)
     *        from a sorted lnbr result vector. Uses @c get_base_vid() as the
     *        @c vertex_id in each dnbr entry so downstream consumers see
     *        base-dataset indices. Trims to @c this->_topk.
     */
    auto _to_knn_results(const std::vector<lnbr_candidate_entry_t>& src) const
        -> knn_results_t
    {
        const std::size_t K = static_cast<std::size_t>(this->_topk);
        knn_results_t out;
        out.reserve(K);
        for (std::size_t i = 0; i < src.size() && out.size() < K; ++i) {
            out.emplace_back(src[i].get_base_vid(), src[i].get_distance());
        }
        while (out.size() < K) {
            out.push_back(dnbr_candidate_entry_t::make_invalid_entry());
        }
        return out;
    }

    /**
     * @brief Warm up the random sequence generator by triggering
     *        thread-local MKL stream creation. Mirrors
     *        @c compact_descent_graph_router.hpp:312-323.
     */
    __attribute__((always_inline))
    auto _warmup_random_seq() const -> void {
        const int num_threads = tbb_max_num_threads();
        tbb::parallel_for(
            tbb::blocked_range<int>(0, num_threads, 1),
            [&](const tbb::blocked_range<int>&) {
                std::vector<vertex_id_t> dummy(1);
                _random_seq.generate(dummy, /*upper_bound=*/1, /*num=*/1);
            }
        );
    }

    // ---- Members ----

    /** @brief Beam width used by high-level query entry points. */
    vertex_num_t _candidate_queue_size = 0;

    /** @brief Pool of thread-local visited bitmaps for parallel beam search. */
    mutable visited_table_pool_t _visited_table_pool;

    /** @brief Thread-safe random sequence generator (MKL-backed). */
    mutable random_seq_t _random_seq;

};  // class InternalGraphRouter

}   // namespace dynamic
}   // namespace cpu
}   // namespace artea
