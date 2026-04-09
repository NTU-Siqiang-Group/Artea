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
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include <artea/common/logger.hpp>
#include <artea/cpu/index/stacked_rgraph/index_structure.hpp>
#include <artea/cpu/router/data_structures/std_candidate_queue.hpp>

namespace artea {
namespace cpu {
namespace stacked_rgraph {

/**
 * @brief Build/insertion algorithm for stacked_rgraph::IndexStructure.
 *
 * Implements the paper's "Approximate r-Net Dynamic Insertion" pseudocode:
 *   Phase 1: Descend through EVERY layer from h_max down to 1, running a
 *            beam search (width @c _search_nn_qs = L_1) at each and caching
 *            the per-layer candidate set C_h. Multi-entry descent carries
 *            the top-K candidates down via @c inter_layer_link.
 *   Phase 1.5: Compute
 *            @c h_p = min{h in [0, h_max - 1] : d_{h+1}* <= R_{h+1}}
 *                     ∪ {h_max + 1}
 *            as the SMALLEST h such that layer (h+1) absorbs p. If no layer
 *            absorbs p, extend the hierarchy by one layer (capped by
 *            @c max_restrict_level).
 *   Phase 2: For h in [1, h_p], insert p at layer h unconditionally. Each
 *            layer's beam search is seeded from the cached Phase 1 C_h at
 *            the same layer and run with the wider @c _select_nbrs_qs
 *            (= L_2) queue to build the ARC-Prune candidate pool. Reverse
 *            edges are added for each chosen neighbor.
 *   Epilogue: Patch every new vertex's inter_layer_link to point at p's
 *            layer_vid in the layer directly below, including the
 *            Phase-1.5 extension vertex (which points at p in h_max).
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
    using internal_graph_t = typename GraphFactoryTraitsT::internal_graph_t;

    // Distance function (inherited via RefinerTraits → ComputerTraits).
    using dist_func_t = typename GraphFactoryTraitsT::dist_func_t;

    // Router-side types (inherited via RefinerTraits → RouterTraits).
    using candidate_queue_t = typename GraphFactoryTraitsT::std_candidate_queue_t;
    using visited_table_t   = typename GraphFactoryTraitsT::visited_table_t;

    static constexpr vertex_id_t invalid_vertex_id = GraphFactoryTraitsT::invalid_vertex_id;

    /** @brief Internal sorted-candidate list type returned by beam search. */
    using candidate_t = std::pair<distance_t, lnbr_t>;

public:
    /**
     * @brief Number of initial vertices inserted serially during bootstrap.
     *        While the hierarchy is still forming, concurrent insertions
     *        tend to create structural conflicts (too many threads trying
     *        to grow the same layer), so we serialize the first
     *        @c startup_points insertions before switching to TBB parallel.
     */
    static constexpr vertex_num_t startup_points = 500;

    /**
     * @brief Build a StackedRGraph from scratch.
     *
     * Construct an @c IndexStructure with the supplied configuration and
     * then call @c add_vertices to insert every vector from @p base_vecs.
     *
     * Returns by @c std::unique_ptr because @c IndexStructure inherits
     * copy/move-deleted from @c HierarchicalGraphV2.
     *
     * @tparam SelectInitialFnT Callable selecting the initial neighbor set
     *         for the new vertex from sorted beam-search candidates.
     *         Signature:
     *         @code
     *         (const vec_ele_t* p_coords,
     *          distance_t R_h,
     *          vertex_num_t max_nbr_size,
     *          const std::vector<std::pair<distance_t, lnbr_t>>& sorted_candidates,
     *          const vector_array_t& base_vecs,
     *          const dist_func_t& dist_func)
     *              -> std::vector<lnbr_t>
     *         @endcode
     * @tparam EvictFnT Callable passed to @c InternalGraph::add_nbr; invoked
     *         when a neighbor array is full to choose which neighbors to
     *         keep. Signature: `(lnbr_t* slots, lnbr_t new_nbr) -> uint64_t`.
     */
    template <typename SelectInitialFnT, typename EvictFnT>
    static auto construct_graph(
        const vector_array_t& base_vecs,
        const dist_func_t& dist_func,
        SelectInitialFnT&& select_initial_fn,
        EvictFnT&& evict_fn,
        const ratio_t rnet_beta,
        const distance_t L1_rnet_radius,
        const vertex_num_t search_nn_qs,
        const vertex_num_t select_nbrs_qs,
        const vertex_num_t max_nbr_size = 32,
        const ratio_t layer_cap_ratio = ratio_t(1),
        const vertex_num_t min_layer_cap = 1000
    ) -> std::unique_ptr<this_index_t> {
        const vertex_num_t total_vertices =
            static_cast<vertex_num_t>(base_vecs.get_num_vecs());
        auto index = std::make_unique<this_index_t>(
            total_vertices,
            rnet_beta, L1_rnet_radius,
            search_nn_qs, select_nbrs_qs,
            max_nbr_size, layer_cap_ratio, min_layer_cap);
        add_vertices(
            *index, base_vecs, dist_func,
            std::forward<SelectInitialFnT>(select_initial_fn),
            std::forward<EvictFnT>(evict_fn));
        return index;
    }

    /**
     * @brief Insert every vector in @p base_vecs as a new vertex of @p index.
     *
     * The first @c startup_points vectors are inserted serially; the rest
     * are inserted via @c tbb::parallel_for. Both phases call the same
     * single-vertex primitive @c _insert_one.
     */
    template <typename SelectInitialFnT, typename EvictFnT>
    static auto add_vertices(
        this_index_t& index,
        const vector_array_t& base_vecs,
        const dist_func_t& dist_func,
        SelectInitialFnT&& select_initial_fn,
        EvictFnT&& evict_fn
    ) -> void {
        const vertex_num_t n = static_cast<vertex_num_t>(base_vecs.get_num_vecs());
        if (n == 0) return;

        // Per-thread reusable visited bitmap: sized to `n` once (the
        // worst-case layer size is the base dataset size). Each beam search
        // call clears and reuses it, avoiding per-call allocation.
        tbb::enumerable_thread_specific<visited_table_t> visited_pool(
            [n]() { return visited_table_t(static_cast<std::size_t>(n)); });

        // Per-thread RNG used for random top-layer sampling in Phase 1.
        // Seeded deterministically from a thread-unique counter so parallel
        // runs produce reproducible (if thread-count-dependent) results.
        tbb::enumerable_thread_specific<std::mt19937> rng_pool(
            []() {
                static std::atomic<uint64_t> seed_ctr{0};
                return std::mt19937(static_cast<uint64_t>(0x9E3779B97F4A7C15ULL)
                                    ^ seed_ctr.fetch_add(1));
            });

        const vertex_num_t cutoff = std::min<vertex_num_t>(startup_points, n);

        // Serial startup.
        {
            auto& visited = visited_pool.local();
            auto& rng = rng_pool.local();
            for (vertex_num_t i = 0; i < cutoff; ++i) {
                _insert_one(index, i, base_vecs, dist_func, select_initial_fn,
                            evict_fn, visited, rng);
            }
        }
        if (cutoff == n) return;

        // Parallel phase.
        tbb::parallel_for(
            tbb::blocked_range<vertex_num_t>(cutoff, n),
            [&](const tbb::blocked_range<vertex_num_t>& r) {
                auto& visited = visited_pool.local();
                auto& rng = rng_pool.local();
                for (vertex_num_t i = r.begin(); i != r.end(); ++i) {
                    _insert_one(index, i, base_vecs, dist_func, select_initial_fn,
                                evict_fn, visited, rng);
                }
            }
        );
    }

private:
    // ============================================================
    //  Single-vertex insertion primitive (the heart of the class)
    // ============================================================

    /**
     * @brief Insert a single base vertex. Thread-safe; this is the only
     *        insertion path used by both serial and parallel phases.
     *
     * @param visited Caller-owned bitmap; reused across insertions.
     * @param rng     Caller-owned RNG; reused across insertions for
     *                the top-layer random seeding.
     */
    template <typename SelectInitialFnT, typename EvictFnT, typename RngT>
    static auto _insert_one(
        this_index_t& index,
        const vertex_id_t p,
        const vector_array_t& base_vecs,
        const dist_func_t& dist_func,
        SelectInitialFnT& select_initial_fn,
        EvictFnT& evict_fn,
        visited_table_t& visited,
        RngT& rng
    ) -> void {
        const vec_ele_t* p_coords = base_vecs.get(p);

        const vertex_num_t search_nn_qs         = index.search_nn_qs();
        const vertex_num_t select_nbrs_qs       = index.select_nbrs_qs();
        const vertex_num_t max_nbr_size         = index.max_nbr_size();
        const layer_num_t  max_restrict_level   = index.max_restrict_level();

        const layer_num_t h_max = index.get_num_layers();

        // ---------- Phase 1: beam-search descent at EVERY layer ----------
        //
        // No entry point is maintained by the hierarchy. The top layer's
        // beam search is seeded directly by drawing up to `search_nn_qs`
        // uniformly random vertices from the top layer itself. Each
        // subsequent layer inherits the previous layer's full candidate
        // set (translated via inter_layer_link) as its entry set.
        //
        //   d_h_stars[h-1] = d_h* (smallest distance at layer h), or +∞ if
        //                    the layer's beam search returned empty.
        //   C_at[h-1]      = full (distance, lnbr_t) candidate set C_h
        //                    (up to search_nn_qs entries, sorted ascending).
        std::vector<distance_t> d_h_stars(
            h_max, std::numeric_limits<distance_t>::max());
        std::vector<std::vector<candidate_t>> C_at(h_max);

        // Reusable buffer: the entry set for the NEXT beam search. At most
        // `search_nn_qs` entries ever carried (one per Phase-1 candidate).
        std::vector<lnbr_t> current_entries;
        current_entries.reserve(static_cast<std::size_t>(search_nn_qs));

        // Seed the top layer's beam search with random top-layer vertices.
        // Only applicable if the hierarchy is non-empty; the empty case is
        // handled by Phase 1.5 below (which will extend to layer 1).
        if (h_max > 0) {
            const auto& top_layer = index.get_layer_graph(h_max - 1);
            const vertex_num_t top_size = top_layer.get_num_vertices();
            if (top_size > 0) {
                if (top_size <= search_nn_qs) {
                    // Take every vertex in the top layer.
                    for (vertex_num_t lv = 0; lv < top_size; ++lv) {
                        current_entries.emplace_back(
                            top_layer.get_base_vid(lv), lv);
                    }
                } else {
                    // Sample `search_nn_qs` distinct vertices without
                    // replacement (rejection sampling; expected iterations
                    // are O(search_nn_qs) when search_nn_qs << top_size).
                    std::unordered_set<vertex_id_t> picked;
                    picked.reserve(static_cast<std::size_t>(search_nn_qs));
                    std::uniform_int_distribution<vertex_num_t>
                        uni_dist(0, top_size - 1);
                    while (picked.size() <
                           static_cast<std::size_t>(search_nn_qs)) {
                        const vertex_id_t rand_lv = uni_dist(rng);
                        if (picked.insert(rand_lv).second) {
                            current_entries.emplace_back(
                                top_layer.get_base_vid(rand_lv), rand_lv);
                        }
                    }
                }
            }
        }

        for (layer_num_t h = h_max; h >= 1; --h) {
            const layer_id_t layer_idx = h - 1;

            auto cands = _beam_search_layer(
                index, p_coords, base_vecs, dist_func, layer_idx,
                std::span<const lnbr_t>(current_entries),
                search_nn_qs, visited);

            if (cands.empty()) {
                // Layer transiently empty (race) or all entries got clamped
                // out. d_h_stars[h-1] stays at +∞ (uncovered); C_at[h-1]
                // stays empty. Clear current_entries — the next layer's
                // beam search will return empty too, and the layer is
                // treated as uncovered.
                if (h > 1) current_entries.clear();
                continue;
            }

            d_h_stars[h - 1] = cands.front().first;

            // Translate the ENTIRE candidate set `cands` to layer (h-1)
            // entries via inter_layer_link BEFORE moving `cands` into the
            // cache. The next layer's beam search inherits the full queue
            // from this layer — no top-K slicing.
            if (h > 1) {
                auto& layer_h = index.get_layer_graph(layer_idx);
                current_entries.clear();
                for (const auto& [d, c] : cands) {
                    (void)d;
                    const vertex_id_t lower_lv =
                        layer_h.get_inter_layer_link(c.layer_vid);
                    current_entries.emplace_back(c.base_vid, lower_lv);
                }
            }

            // Cache the full Phase 1 candidate set C_h for Phase 2 reuse.
            C_at[h - 1] = std::move(cands);
        }

        // ---------- Compute h_p ----------
        //
        //   h_p = min{h in [0, h_max - 1] : d_{h+1}* ≤ R_{h+1}} ∪ {h_max + 1}
        //
        // h_p is the SMALLEST h such that layer (h+1) already absorbs p.
        // Layers [h_p + 1 .. h_max] are deemed covered (even if a later
        // layer was uncovered — we stop at the first found cover). Layers
        // [1 .. h_p] are deemed uncovered and get an insertion in Phase 2.
        // If no layer covers p, h_p = h_max + 1 and we extend the hierarchy.
        layer_num_t h_p = h_max + 1;
        for (layer_num_t h = 0; h < h_max; ++h) {
            if (d_h_stars[h] <= index.radius_at(h + 1)) {
                h_p = h;
                break;
            }
        }

        // ---------- Phase 1.5: extend hierarchy if uncovered everywhere ----------
        //
        // Extension is capped by `max_restrict_level`. If the cap is
        // reached we simply set h_p := h_max and proceed (p will still be
        // inserted at every existing layer, but no new layer is created).
        //
        // `extend_and_seed` runs the seed lambda under its internal mutex
        // and publishes the new layer only AFTER the seed vertex is in
        // place — so concurrent readers never see an empty new top layer.
        // It returns `true` if growth actually happened, or `false` if
        // another thread had already grown past our target.
        std::optional<std::pair<layer_id_t, vertex_id_t>> ext_slot;
        if (h_p == h_max + 1) {
            const layer_num_t new_h = h_max + 1;
            if (new_h > max_restrict_level) {
                h_p = h_max;
            } else {
                const vertex_num_t base_n =
                    static_cast<vertex_num_t>(base_vecs.get_num_vecs());
                index.extend_and_seed(
                    new_h,
                    [&](const layer_id_t new_layer_id) {
                        const vertex_num_t cap =
                            index.capacity_for_layer(new_layer_id, base_n);
                        return std::make_unique<internal_graph_t>(
                            cap, max_nbr_size);
                    },
                    [&](const layer_id_t top_idx,
                        internal_graph_t& top_layer) -> void {
                        // Placeholder inter_layer_link (= p). Patched by
                        // the Phase-2 epilogue once we know p's layer_vid
                        // in the layer directly below. If h_max was 0
                        // (first-ever insertion), there is nothing to
                        // patch and the placeholder stays — which is the
                        // correct L0 identity (base_vid == p).
                        const vertex_id_t new_top_lv = top_layer.add_vertex(p);
                        ext_slot = std::make_pair(top_idx, new_top_lv);
                    });
                h_p = h_max;
            }
        }

        // ---------- Phase 2: insert p at layers [1, h_p] ----------
        if (h_p == 0) {
            // p is covered at layer 1 already → no insertion in upper hierarchy.
            return;
        }

        // Reusable scratch buffer for converting a cached C_h into lnbr_t
        // entries for _beam_search_layer.
        std::vector<lnbr_t> phase2_entries;
        phase2_entries.reserve(static_cast<std::size_t>(search_nn_qs) + 1);

        std::vector<std::pair<layer_id_t, vertex_id_t>> insertions;
        insertions.reserve(h_p);

        for (layer_num_t h = h_p; h >= 1; --h) {
            const layer_id_t layer_idx = h - 1;
            auto& layer_h = index.get_layer_graph(layer_idx);
            const distance_t R_h = index.radius_at(h);

            // Seed Phase 2's beam search from the cached Phase 1 candidate
            // set C_h at the SAME layer. We do not propagate phase2_entries
            // down via inter_layer_link between Phase 2 iterations; every
            // layer starts fresh from its own in-layer Phase 1 output. If
            // Phase 1's C_h was empty (transient race), we leave
            // phase2_entries empty — the beam search will return no
            // candidates and we'll skip neighbor selection for that layer.
            phase2_entries.clear();
            if (h - 1 < h_max && !C_at[h - 1].empty()) {
                for (const auto& [d, lv] : C_at[h - 1]) {
                    (void)d;
                    phase2_entries.push_back(lv);
                }
            }

            // Rich beam search with select_nbrs_qs (= L_2) for ARC-Prune.
            auto candidates = _beam_search_layer(
                index, p_coords, base_vecs, dist_func, layer_idx,
                std::span<const lnbr_t>(phase2_entries),
                select_nbrs_qs, visited);

            // Insert p at this layer. Placeholder lower-layer vid is `p`
            // (the base_vid); upper layers will be patched in the epilogue.
            const vertex_id_t new_lv_h = layer_h.add_vertex(p);
            insertions.emplace_back(layer_idx, new_lv_h);
            const lnbr_t p_lnbr_h(p, new_lv_h);

            auto initial_nbrs = select_initial_fn(
                p_coords, R_h, max_nbr_size, candidates, base_vecs, dist_func);

            // Forward edges.
            for (const lnbr_t& n : initial_nbrs) {
                layer_h.add_nbr(new_lv_h, n, evict_fn);
            }
            // Reverse edges.
            for (const lnbr_t& n : initial_nbrs) {
                layer_h.add_nbr(n.layer_vid, p_lnbr_h, evict_fn);
            }
        }

        // ---------- Phase 2 epilogue: patch inter-layer links ----------
        // insertions[0] is the topmost (h = h_p), insertions.back() is h = 1.
        // For each upper layer h ≥ 2, set its new vertex's inter_layer_link
        // to p's new layer_vid in layer h - 1. Layer 1's placeholder link is
        // `p` (= base_vid), which is already the correct L0 identity.
        for (std::size_t i = 0; i + 1 < insertions.size(); ++i) {
            const auto [upper_idx, upper_lv] = insertions[i];
            const auto [lower_idx, lower_lv] = insertions[i + 1];
            (void)lower_idx;
            auto& upper_layer = index.get_layer_graph(upper_idx);
            upper_layer.set_inter_layer_link(upper_lv, lower_lv);
        }

        // Patch the Phase-1.5 extension vertex's inter_layer_link to point
        // at p's layer_vid in layer h_max. `insertions.front()` is the
        // topmost Phase 2 layer (h = h_p = h_max), so its new_lv is p's
        // identity in that layer.
        if (ext_slot && !insertions.empty()) {
            auto& ext_layer = index.get_layer_graph(ext_slot->first);
            ext_layer.set_inter_layer_link(
                ext_slot->second, insertions.front().second);
        }
    }

    // ============================================================
    //  Single-layer search helper
    // ============================================================

    /**
     * @brief Beam search on a single layer starting from a set of @p entries.
     *        Returns up to @p queue_size (distance, lnbr_t) pairs sorted by
     *        ascending distance.
     *
     * Uses the router's @c StdCandidateQueue (via the internal
     * @c _queue_traits adapter) plus a caller-provided @c ThreadLocalBitmap
     * for visited tracking. The bitmap is reused across calls (see
     * @c add_vertices); the function calls @c visited.clear() on entry.
     *
     * Clamps each entry's @c layer_vid against the current layer size and
     * clamps the raw neighbor count against @c max_nbr_size to defend
     * against stale/torn reads during concurrent extension.
     *
     * Reads are lock-free: the per-vertex spinlock is NOT acquired. Stale
     * reads of newly-inserted neighbors are tolerated (the algorithm is
     * designed to work under optimistic concurrency per the paper).
     */
    static auto _beam_search_layer(
        const this_index_t& index,
        const vec_ele_t* query_coords,
        const vector_array_t& base_vecs,
        const dist_func_t& dist_func,
        const layer_id_t layer_idx,
        std::span<const lnbr_t> entries,
        const vertex_num_t queue_size,
        visited_table_t& visited
    ) -> std::vector<candidate_t> {
        const auto& layer = index.get_layer_graph(layer_idx);
        const vertex_num_t layer_size = layer.get_num_vertices();
        if (layer_size == 0) return {};

        const vertex_num_t max_nbr = layer.max_nbr_size();
        const std::size_t L = static_cast<std::size_t>(queue_size);

        visited.clear();
        candidate_queue_t cq(L);

        // Side map: layer_vid -> base_vid. Needed because CandidateQueue
        // stores only the 31-bit vertex_id (we use layer_vid there for
        // visited-tracking efficiency), but we need to return full
        // (base_vid, layer_vid) lnbr_t pairs. We only touch at most
        // `queue_size * max_nbr` entries, so a small hash map wins over
        // a layer-sized vector.
        std::unordered_map<vertex_id_t, vertex_id_t> lv_to_bv;
        lv_to_bv.reserve(static_cast<std::size_t>(L) * 8);

        // Seed with every (deduped, in-bounds) entry point.
        for (const lnbr_t& e : entries) {
            if (e.base_vid == invalid_vertex_id) continue;
            const vertex_id_t lv = e.layer_vid;
            if (lv >= layer_size) continue;  // stale entry, skip
            if (visited.test_and_set(lv)) continue;
            lv_to_bv.emplace(lv, e.base_vid);
            const distance_t d =
                dist_func(query_coords, base_vecs.get(e.base_vid));
            cq.try_push(lv, d);
        }

        if (cq.empty()) return {};

        // Standard beam-search loop (matches the router pattern).
        while (!cq.empty()) {
            if (cq.should_terminate()) break;
            const auto [cur_lv, cur_dist] = cq.pop_best_unexplored();
            if (cur_lv == invalid_vertex_id) break;

            const auto block = layer.fetch_nbrs(cur_lv);
            const uint64_t raw_count = layer.num_valid_nbrs(cur_lv);
            const uint64_t count = std::min<uint64_t>(raw_count, max_nbr);
            for (uint64_t i = 0; i < count; ++i) {
                const lnbr_t nbr = block[1 + i];
                if (nbr.base_vid == invalid_vertex_id) continue;
                if (nbr.layer_vid >= layer_size) continue;  // torn read
                if (visited.test_and_set(nbr.layer_vid)) continue;
                lv_to_bv.emplace(nbr.layer_vid, nbr.base_vid);
                const distance_t d =
                    dist_func(query_coords, base_vecs.get(nbr.base_vid));
                cq.try_push(nbr.layer_vid, d);
            }
        }

        // Extract results (sorted ascending). Clamp k to the number of
        // results actually held, since extract_results asserts size() >= k.
        const std::size_t k = std::min(L, cq.get_result_size());
        auto entries_out = cq.extract_results(k);

        std::vector<candidate_t> out;
        out.reserve(entries_out.size());
        for (const auto& entry : entries_out) {
            const vertex_id_t lv = entry.get_id();
            const auto it = lv_to_bv.find(lv);
            const vertex_id_t bv =
                (it != lv_to_bv.end()) ? it->second : invalid_vertex_id;
            out.emplace_back(entry.distance, lnbr_t(bv, lv));
        }
        return out;
    }
};

}   // namespace stacked_rgraph
}   // namespace cpu
}   // namespace artea
