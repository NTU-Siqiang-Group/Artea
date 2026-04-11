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
    using lnbr_t           = typename GraphFactoryTraitsT::lnbr_t;
    using vector_array_t   = typename GraphFactoryTraitsT::vector_array_t;
    using internal_graph_t = typename GraphFactoryTraitsT::dynamic::internal_graph_t;

    // Distance function (inherited via RefinerTraits → ComputerTraits).
    using dist_func_t = typename GraphFactoryTraitsT::dist_func_t;

    // Router-side types (inherited via RefinerTraits → RouterTraits).
    using visited_table_t         = typename GraphFactoryTraitsT::visited_table_t;
    using lnbr_candidate_entry_t  = typename GraphFactoryTraitsT::lnbr_candidate_entry_t;
    using hg_router_t             = typename GraphFactoryTraitsT::dynamic::hierarchical_graph_router_t;

    static constexpr vertex_id_t invalid_vertex_id = GraphFactoryTraitsT::invalid_vertex_id;

    /** @brief Internal sorted-candidate list type returned by beam search. */
    using candidate_t = lnbr_candidate_entry_t;

public:
    /**
     * @brief Number of initial vertices inserted serially during bootstrap.
     *        While the hierarchy is still forming, concurrent per_level_insertions
     *        tend to create structural conflicts (too many threads trying
     *        to grow the same layer), so we serialize the first
     *        @c startup_points per_level_insertions before switching to TBB parallel.
     */
    static constexpr vertex_num_t startup_points = 500;

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
        // (beam_search_layer); no query-path warmup is needed because
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
     * @c router.sample_random_entries, which owns its own thread-safe
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
        std::vector<std::vector<candidate_t>> candidates_per_layer(cur_max_level);

        // Reusable buffer: the entry set for the NEXT beam search. At most
        // `search_nn_qs` entries ever carried (one per Phase-1 candidate).
        std::vector<lnbr_t> next_layer_seeds;
        next_layer_seeds.reserve(static_cast<std::size_t>(search_nn_qs));

        // Seed the top layer's beam search with random top-layer vertices.
        // Delegated to the router so that the factory, the query-path
        // beam_search, and the per-layer router all share a single
        // implementation. Empty hierarchy is handled by Phase 1.5 below
        // (which will extend to layer 1).
        if (cur_max_level > 0) {
            const auto& top_layer = index.get_layer_graph(cur_max_level - 1);
            router.sample_random_entries(top_layer, search_nn_qs, next_layer_seeds);
        }

        for (layer_num_t cur_level = cur_max_level; cur_level >= 1; --cur_level) {
            const layer_id_t cur_layer_idx = cur_level - 1;

            auto cur_layer_cands = router.beam_search_layer(
                new_vec,
                index.get_layer_graph(cur_layer_idx),
                std::span<const lnbr_t>(next_layer_seeds),
                search_nn_qs, visited);

            if (cur_layer_cands.empty()) {
                // Layer transiently empty (race) or all entries got clamped
                // out. min_distance_per_layer[cur_level-1] stays at +∞
                // (uncovered); candidates_per_layer[cur_level-1] stays
                // empty. Clear next_layer_seeds — the next layer's beam
                // search will return empty too, and the layer is treated as
                // uncovered.
                if (cur_level > 1) next_layer_seeds.clear();
                continue;
            }

            min_distance_per_layer[cur_level - 1] = cur_layer_cands.front().get_distance();

            // Translate the ENTIRE candidate set `cur_layer_cands` to layer
            // (cur_level - 1) entries via inter_layer_link BEFORE moving it
            // into the cache. The next layer's beam search inherits the
            // full queue from this layer — no top-K slicing.
            if (cur_level > 1) {
                auto& cur_layer_graph = index.get_layer_graph(cur_layer_idx);
                next_layer_seeds.clear();
                for (const auto& cand : cur_layer_cands) {
                    const vertex_id_t lower_layer_vid =
                        cur_layer_graph.get_inter_layer_link(cand.get_layer_vid());
                    next_layer_seeds.emplace_back(cand.get_base_vid(), lower_layer_vid);
                }
            }

            // Cache the full Phase 1 candidate set for Phase 2 reuse.
            candidates_per_layer[cur_level - 1] = std::move(cur_layer_cands);
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

        // Reusable scratch buffer for converting a cached Phase-1
        // candidate list into lnbr_t entries passed to
        // router.beam_search_layer.
        std::vector<lnbr_t> cand_pool_seeds;
        cand_pool_seeds.reserve(static_cast<std::size_t>(search_nn_qs) + 1);

        std::vector<std::pair<layer_id_t, vertex_id_t>> per_level_insertions;
        per_level_insertions.reserve(highest_insert_level);

        // Scratch buffer for the pruned initial neighbor set. Sized once
        // at the upper bound (max_nbr_size); reused across Phase 2 layers.
        std::vector<lnbr_t> selected_nbrs;
        selected_nbrs.reserve(static_cast<std::size_t>(max_nbr_size));

        for (layer_num_t cur_level = highest_insert_level; cur_level >= 1; --cur_level) {
            const layer_id_t cur_layer_idx = cur_level - 1;
            auto& cur_layer_graph = index.get_layer_graph(cur_layer_idx);

            // Seed Phase 2's beam search from the cached Phase 1 candidate
            // set at the SAME layer. We do not propagate cand_pool_seeds down
            // via inter_layer_link between Phase 2 iterations; every layer
            // starts fresh from its own in-layer Phase 1 output. If Phase
            // 1's candidates_per_layer[cur_level - 1] was empty (transient
            // race), we leave cand_pool_seeds empty — the beam search will
            // return no candidates and we'll skip neighbor selection for
            // that layer.
            cand_pool_seeds.clear();
            if (cur_level - 1 < cur_max_level && !candidates_per_layer[cur_level - 1].empty()) {
                for (const auto& cand : candidates_per_layer[cur_level - 1]) {
                    cand_pool_seeds.emplace_back(cand.get_base_vid(), cand.get_layer_vid());
                }
            }

            // Rich beam search with select_nbrs_qs (= L_2) to build a
            // larger-than-max_nbr_size candidate pool.
            auto cur_layer_cands = router.beam_search_layer(
                new_vec, cur_layer_graph,
                std::span<const lnbr_t>(cand_pool_seeds),
                select_nbrs_qs, visited);

            // Insert the new vertex at this layer. Placeholder lower-layer
            // vid is new_base_vid (the base_vid); upper layers will be patched
            // in the epilogue.
            const vertex_id_t new_layer_vid = cur_layer_graph.add_vertex(new_base_vid);
            per_level_insertions.emplace_back(cur_layer_idx, new_layer_vid);
            const lnbr_t new_vertex_lnbr(new_base_vid, new_layer_vid);

            // Neighbor selection (pruning): the beam-search output is
            // already sorted ascending by distance. Simply truncate to at
            // most max_nbr_size entries. No covering-radius filter, no
            // dominance pruning — the construction parameters (select_nbrs_qs,
            // max_nbr_size) already encode the desired trade-off.
            selected_nbrs.clear();
            const std::size_t num_to_take = std::min<std::size_t>(
                cur_layer_cands.size(), static_cast<std::size_t>(max_nbr_size));
            for (std::size_t i = 0; i < num_to_take; ++i) {
                selected_nbrs.emplace_back(
                    cur_layer_cands[i].get_base_vid(),
                    cur_layer_cands[i].get_layer_vid());
            }

            // Forward edges.
            for (const lnbr_t& nbr : selected_nbrs) {
                cur_layer_graph.add_nbr(new_layer_vid, nbr, pruning_fn);
            }
            // Reverse edges.
            for (const lnbr_t& nbr : selected_nbrs) {
                cur_layer_graph.add_nbr(nbr.layer_vid, new_vertex_lnbr, pruning_fn);
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
