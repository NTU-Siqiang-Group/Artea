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
 * @FilePath: /Artea/include/artea/cpu/index/stacked_rgraph/index_factory.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Build algorithm for the dynamic Stacked R-Net index.
 *               Operates on a stacked_rgraph::IndexStructure via static
 *               methods — all state lives in the IndexStructure.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include <artea/common/logger.hpp>

namespace artea {
namespace cpu {
namespace stacked_rgraph {

/**
 * @brief Build/insertion algorithm for stacked_rgraph::IndexStructure.
 *
 * Implements the paper's "Approximate r-Net Dynamic Insertion" pseudocode:
 *   Phase 1: Descend through EVERY layer from cur_max_level down to 1,
 *            running a beam search (width @c _search_nn_qs = L_1) at each
 *            and caching the per-layer candidate set. Multi-entry descent
 *            carries the top-K candidates down via @c inter_layer_link.
 *   Phase 1.5: Compute @c highest_insert_level as the SMALLEST level such that
 *            the layer directly above already absorbs the new vertex. If
 *            no layer covers the new vertex, extend the hierarchy by one
 *            layer (capped by @c max_restrict_level).
 *   Phase 2: For every level in [1, highest_insert_level], insert the new
 *            vertex unconditionally. Each layer's beam search is seeded
 *            from the cached Phase 1 candidate set at the same layer and
 *            run with the wider @c _select_nbrs_qs (= L_2) queue to build
 *            the candidate pool for neighbor pruning. Reverse edges are
 *            added for each chosen neighbor.
 *   Epilogue: Patch every new vertex's inter_layer_link to point at its
 *            layer_vid in the layer directly below, including the
 *            Phase-1.5 extension vertex.
 *
 * @tparam GraphFactoryTraitsT The graph factory traits type.
 */
template <typename GraphFactoryTraitsT>
class IndexFactory {

    using this_index_t     = typename GraphFactoryTraitsT::stacked_rgraph::index_t;

    using vertex_num_t     = typename GraphFactoryTraitsT::vertex_num_t;
    using vertex_id_t      = typename GraphFactoryTraitsT::vertex_id_t;
    using layer_num_t      = typename GraphFactoryTraitsT::layer_num_t;
    using layer_id_t       = typename GraphFactoryTraitsT::layer_id_t;
    using distance_t       = typename GraphFactoryTraitsT::distance_t;
    using ratio_t          = typename GraphFactoryTraitsT::ratio_t;
    using vec_ele_t        = typename GraphFactoryTraitsT::vec_ele_t;
    using inbr_t           = typename GraphFactoryTraitsT::inbr_t;
    using vector_array_t   = typename GraphFactoryTraitsT::vector_array_t;
    using internal_graph_t = typename GraphFactoryTraitsT::dynamic::internal_graph_t;

    // Distance function (inherited via RefinerTraits → ComputerTraits).
    using dist_func_t = typename GraphFactoryTraitsT::dist_func_t;

    // Router-side types (inherited via RefinerTraits → RouterTraits).
    using visited_table_t         = typename GraphFactoryTraitsT::visited_table_t;
    using candidate_entry_t       = typename GraphFactoryTraitsT::candidate_entry_t;
    using std_candidate_queue_t   = typename GraphFactoryTraitsT::std_candidate_queue_t;
    using knn_results_t           = typename GraphFactoryTraitsT::knn_results_t;
    using hg_router_t             = typename GraphFactoryTraitsT::dynamic::hierarchical_graph_router_t;

    static constexpr vertex_id_t invalid_vertex_id = GraphFactoryTraitsT::invalid_vertex_id;

public:
    /**
     * @brief Number of initial vertices inserted serially during bootstrap.
     *        While the hierarchy is still forming, concurrent per_level_insertions
     *        tend to create structural conflicts (too many threads trying
     *        to grow the same layer), so we serialize the first
     *        @c startup_points per_level_insertions before switching to TBB parallel.
     */
    static constexpr vertex_num_t startup_points = 100;

    /**
     * @brief Append @p batch_vecs to @p index's owned vector storage and
     *        insert every newly-appended vector as a new vertex of the
     *        hierarchy.
     *
     * This is the primary incremental entry point. On every call:
     *   1. @p batch_vecs is appended to @c index.get_vecs_storage() via
     *      @c IndexStructure::append_vecs. If the index's owned storage
     *      was empty this is a move; otherwise it is a parallel copy.
     *      After this step @p batch_vecs is left in whatever state
     *      @c append_batch leaves it.
     *   2. A @c hg_router_t is constructed locally over the freshly-grown
     *      vector storage — the router's @c _visited_table_pool is
     *      sized to the CURRENT total, so incremental calls always see
     *      a correctly-sized router.
     *   3. The newly-inserted vertices, occupying @c base_vid range
     *      @c [base_vid_offset, base_vid_offset + batch_size), are fed
     *      through @c _insert_one. The first @c startup_points vertices
     *      are inserted serially; the remainder run under
     *      @c tbb::parallel_for.
     *
     * Note that @c base_vid values passed to @c _insert_one are indices
     * into @c index.get_vecs_storage() — NOT into @p batch_vecs. This
     * is why the inner loop variable is called @c base_vid (not @c bvid):
     * every call site refers to the owned storage.
     */
    template <typename PruningFnT>
    static auto add_vertices(
        this_index_t& index,
        vector_array_t&& batch_vecs,
        const dist_func_t& dist_func,
        PruningFnT&& pruning_fn
    ) -> void {
        const vertex_num_t batch_size =
            static_cast<vertex_num_t>(batch_vecs.get_num_vecs());
        if (batch_size == 0) return;

        // Append the batch into the index's owned vector storage. The
        // new vertices occupy base_vid range
        // [base_vid_offset, base_vid_offset + batch_size).
        const vertex_num_t base_vid_offset =
            static_cast<vertex_num_t>(index.get_vecs_storage().get_num_vecs());
        index.append_vecs(std::move(batch_vecs));

        // From here on, all base_vid values reference the OWNED storage.
        const auto& vecs_storage = index.get_vecs_storage();
        const vertex_num_t total_vecs_storage =
            static_cast<vertex_num_t>(vecs_storage.get_num_vecs());
        const vertex_num_t base_vid_end = base_vid_offset + batch_size;

        // Router is constructed internally, over the up-to-date vector
        // storage. It is used only via its build-time primitive
        // (beam_search); no query-path warmup is needed because
        // we pass the factory's own visited table on the build path.
        // The router's `topk` is irrelevant for the atom;
        // select_nbrs_qs is passed as a harmless placeholder.
        hg_router_t router(
            vecs_storage, dist_func,
            /*topk=*/std::max(index.search_nn_qs(), index.select_nbrs_qs()),
            /*search_nn_qs=*/index.search_nn_qs(),
            /*candidate_queue_size=*/index.select_nbrs_qs());

        // Per-thread reusable visited bitmap: sized to total_vecs_storage
        // (the worst-case layer size equals the whole base dataset).
        // Each beam search call clears and reuses it, avoiding per-call
        // allocation.
        tbb::enumerable_thread_specific<visited_table_t> visited_pool(
            [total_vecs_storage]() {
                return visited_table_t(
                    static_cast<std::size_t>(total_vecs_storage));
            });

        const vertex_num_t serial_cutoff =
            base_vid_offset +
            std::min<vertex_num_t>(startup_points, batch_size);

        // Serial startup.
        {
            auto& visited = visited_pool.local();
            for (vertex_num_t base_vid = base_vid_offset;
                 base_vid < serial_cutoff; ++base_vid) {
                _insert_one(index, router, base_vid, dist_func,
                            pruning_fn, visited);
            }
        }
        if (serial_cutoff == base_vid_end) return;

        // Parallel phase.
        tbb::parallel_for(
            tbb::blocked_range<vertex_num_t>(serial_cutoff, base_vid_end),
            [&](const tbb::blocked_range<vertex_num_t>& r) {
                auto& visited = visited_pool.local();
                for (vertex_num_t base_vid = r.begin();
                     base_vid != r.end(); ++base_vid) {
                    _insert_one(index, router, base_vid, dist_func,
                                pruning_fn, visited);
                }
            }
        );

        // Remove trailing layers that received no vertices.
        index.trim_empty_layers();
    }

private:
    // ============================================================
    //  Single-vertex insertion primitive (the heart of the class)
    // ============================================================

    /**
     * @brief Insert a single base vertex. Thread-safe; this is the only
     *        insertion path used by both serial and parallel phases.
     *
     * @c new_base_vid indexes into @c index.get_vecs_storage() — the
     * owned vector storage, NOT any external batch. Callers must have
     * already appended the batch into @p index via
     * @c IndexStructure::append_vecs before invoking this method.
     *
     * Top-layer random seeding is delegated to
     * @c router.sample_entries, which owns its own thread-safe
     * MKL-backed RNG (TBB enumerable_thread_specific) — no separate
     * RNG needs to be threaded through.
     *
     * @param visited Caller-owned bitmap; reused across insertions.
     */
    template <typename PruningFnT>
    static auto _insert_one(
        this_index_t& index,
        const hg_router_t& router,
        const vertex_id_t new_base_vid,
        const dist_func_t& dist_func,
        PruningFnT& pruning_fn,
        visited_table_t& visited
    ) -> void {
        const auto& vecs_storage = index.get_vecs_storage();
        const vec_ele_t* new_vec = vecs_storage.get(new_base_vid);

        const vertex_num_t search_nn_qs         = index.search_nn_qs();
        const vertex_num_t select_nbrs_qs       = index.select_nbrs_qs();
        const vertex_num_t max_nbr_size         = index.max_nbr_size();
        const layer_num_t  max_restrict_level   = index.max_restrict_level();

        const layer_num_t cur_max_level = index.get_num_layers();

        // ---------- Phase 1: beam-search descent at EVERY layer ----------
        //
        // No entry point is maintained by the hierarchy. The top layer's
        // beam search is seeded directly by drawing up to `search_nn_qs`
        // uniformly random vertices from the top layer itself. Each
        // subsequent layer inherits the previous layer's full candidate
        // set (translated via inter_layer_link) as its entry set.
        //
        //   min_distance_per_layer[cur_level - 1] = smallest candidate
        //        distance seen at cur_level, or +∞ if the layer's beam
        //        search returned empty.
        //   candidates_per_layer[cur_level - 1]   = the full sorted
        //        candidate set from the layer's beam search (up to
        //        search_nn_qs entries, ascending by distance).
        std::vector<distance_t> min_distance_per_layer(
            cur_max_level, std::numeric_limits<distance_t>::max());
        std::vector<std::unique_ptr<std_candidate_queue_t>> candidates_per_layer(cur_max_level);

        const std::size_t L1 = static_cast<std::size_t>(search_nn_qs);

        // Seed the top layer's beam search with random top-layer vertices.
        std_candidate_queue_t candidate_queue(L1);
        if (cur_max_level > 0) {
            const auto& top_layer = index.get_layer_graph(cur_max_level - 1);
            router.sample_entries(top_layer, new_vec, candidate_queue);
        }

        for (layer_num_t cur_level = cur_max_level; cur_level >= 1; --cur_level) {
            const layer_id_t cur_layer_idx = cur_level - 1;

            router.beam_search(
                new_vec, index.get_layer_graph(cur_layer_idx), candidate_queue, visited);

            if (candidate_queue.get_result_size() == 0) {
                // Empty queue — create a fresh empty one for next layer.
                if (cur_level > 1) candidate_queue = std_candidate_queue_t(L1);
                continue;
            }

            // Find min distance by iterating the queue (heap order is fine).
            distance_t min_d = std::numeric_limits<distance_t>::max();
            for (const auto& c : candidate_queue) {
                if (c.get_distance() < min_d) min_d = c.get_distance();
            }
            min_distance_per_layer[cur_level - 1] = min_d;

            // Cache via clone() for Phase 2 reuse.
            candidates_per_layer[cur_level - 1] =
                std::make_unique<std_candidate_queue_t>(candidate_queue.clone());

            // Translate to next layer via inter_layer_link.
            if (cur_level > 1) {
                auto& cur_layer_graph = index.get_layer_graph(cur_layer_idx);
                std_candidate_queue_t next_candidate_queue(L1);
                for (const auto& cand : candidate_queue) {
                    const vertex_id_t lower_layer_vid =
                        cur_layer_graph.get_inter_layer_link(cand.get_layer_vid());
                    next_candidate_queue.try_push(cand.get_base_vid(), lower_layer_vid,
                                     cand.get_distance());
                }
                candidate_queue = std::move(next_candidate_queue);
            }
        }

        // ---------- Compute highest_insert_level ----------
        //
        //   highest_insert_level = the SMALLEST level in [0, cur_max_level - 1]
        //       such that layer (level + 1) already absorbs the new vertex
        //       (i.e. the best distance at that layer is within the layer's
        //       covering radius). Sentinel value: cur_max_level + 1.
        //
        // Layers [highest_insert_level + 1 .. cur_max_level] are deemed covered
        // (even if a later layer was uncovered — we stop at the first found
        // cover). Layers [1 .. highest_insert_level] are deemed uncovered and
        // get an insertion in Phase 2. If no layer covers new_base_vid,
        // highest_insert_level = cur_max_level + 1 and we extend the hierarchy.
        layer_num_t highest_insert_level = cur_max_level + 1;
        for (layer_num_t cur_level = 0; cur_level < cur_max_level; ++cur_level) {
            if (min_distance_per_layer[cur_level] <= index.radius_at(cur_level + 1)) {
                highest_insert_level = cur_level;
                break;
            }
        }

        // ---------- Phase 1.5: cap highest_insert_level ----------
        //
        // All layers are pre-allocated at construction. If no layer
        // covers the new vertex, insert it at every layer up to
        // max_restrict_level.
        if (highest_insert_level > max_restrict_level) {
            highest_insert_level = max_restrict_level;
        }

        // ---------- Phase 2: insert new_base_vid at layers [1, highest_insert_level] ----------
        if (highest_insert_level == 0) {
            // new_base_vid is already covered at layer 1 → no insertion in
            // upper hierarchy.
            return;
        }

        const std::size_t L2 = static_cast<std::size_t>(select_nbrs_qs);

        std::vector<std::pair<layer_id_t, vertex_id_t>> per_level_insertions;
        per_level_insertions.reserve(highest_insert_level);

        // Scratch buffer for the pruned initial neighbor set. Sized once
        // at the upper bound (max_nbr_size); reused across Phase 2 layers.
        std::vector<inbr_t> selected_nbrs;
        selected_nbrs.reserve(static_cast<std::size_t>(max_nbr_size));

        for (layer_num_t cur_level = highest_insert_level; cur_level >= 1; --cur_level) {
            const layer_id_t cur_layer_idx = cur_level - 1;
            auto& cur_layer_graph = index.get_layer_graph(cur_layer_idx);

            // Seed Phase 2's beam search from the cached Phase 1 candidate
            // set at the SAME layer (wider capacity L2 for richer search).
            std_candidate_queue_t phase2_candidate_queue(L2);
            if (cur_level - 1 < cur_max_level && candidates_per_layer[cur_level - 1]) {
                for (const auto& cand : *candidates_per_layer[cur_level - 1]) {
                    phase2_candidate_queue.try_push(cand.get_base_vid(),
                                       cand.get_layer_vid(),
                                       cand.get_distance());
                }
            }

            // Rich beam search with select_nbrs_qs (= L_2).
            router.beam_search(new_vec, cur_layer_graph, phase2_candidate_queue, visited);

            // Insert the new vertex at this layer.
            const vertex_id_t new_layer_vid = cur_layer_graph.add_vertex(new_base_vid);
            per_level_insertions.emplace_back(cur_layer_idx, new_layer_vid);
            const inbr_t new_vertex_inbr(new_base_vid, new_layer_vid);

            // Neighbor selection: extract sorted results, truncate to max_nbr_size.
            const std::size_t result_k = std::min(
                L2, phase2_candidate_queue.get_result_size());
            auto cur_layer_cands = (result_k > 0)
                ? phase2_candidate_queue.extract_results(result_k)
                : knn_results_t{};

            selected_nbrs.clear();
            const std::size_t num_to_take = std::min<std::size_t>(
                cur_layer_cands.size(), static_cast<std::size_t>(max_nbr_size));
            for (std::size_t i = 0; i < num_to_take; ++i) {
                selected_nbrs.emplace_back(
                    cur_layer_cands[i].get_base_vid(),
                    cur_layer_cands[i].get_layer_vid());
            }

            // Forward edges.
            for (const inbr_t& nbr : selected_nbrs) {
                cur_layer_graph.add_nbr(new_layer_vid, nbr, pruning_fn);
            }
            // Reverse edges.
            for (const inbr_t& nbr : selected_nbrs) {
                cur_layer_graph.add_nbr(nbr.get_level_vid(), new_vertex_inbr, pruning_fn);
            }
        }

        // ---------- Phase 2 epilogue: patch inter-layer links ----------
        // per_level_insertions[0] is the topmost (cur_level = highest_insert_level);
        // per_level_insertions.back() is cur_level = 1. For every upper
        // layer (cur_level ≥ 2), set the new vertex's inter_layer_link to
        // its new layer_vid in the layer directly below. Layer 1's
        // placeholder link is new_base_vid (= base_vid), which is already the
        // correct L0 identity.
        for (std::size_t i = 0; i + 1 < per_level_insertions.size(); ++i) {
            const auto [upper_layer_idx, upper_layer_vid] = per_level_insertions[i];
            const auto [lower_layer_idx, lower_layer_vid] = per_level_insertions[i + 1];
            (void)lower_layer_idx;
            auto& upper_layer_graph = index.get_layer_graph(upper_layer_idx);
            upper_layer_graph.set_inter_layer_link(upper_layer_vid, lower_layer_vid);
        }

    }

};

}   // namespace stacked_rgraph
}   // namespace cpu
}   // namespace artea
