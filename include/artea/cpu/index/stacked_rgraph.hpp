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
 * @FilePath: /Artea/include/artea/cpu/index/stacked_rgraph.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Dynamic Stacked R-Nets index built on top of HierarchicalGraphV2.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include <artea/common/logger.hpp>
#include <artea/cpu/containers/thread_local_bitmap.hpp>
#include <artea/cpu/index/hierarchical_graph_v2.hpp>
#include <artea/cpu/router/data_structures/candidate_entry.hpp>
#include <artea/cpu/router/data_structures/std_candidate_queue.hpp>

namespace artea {
namespace cpu {

/**
 * @brief Dynamic hierarchical r-net structure built via online single-vertex
 *        insertion (HNSW-style). Inherits from @c HierarchicalGraphV2 and
 *        manages the upper layers L1..L_max; the base layer L0 is the
 *        caller's full @c vector_array_t and is not stored by this class.
 *
 * Each upper-layer vertex stores its dual identity via @c lnbr_t
 * (@c base_vid is the position in the caller's base_vecs; @c layer_vid is
 * the position in the layer's @c InternalGraph).
 *
 * Per-layer covering radius: @c R_h = L1_rnet_radius * rnet_beta^(h-1), with
 * @c h = 1 being the lowest upper layer (layer_id == 0 internally).
 *
 * @tparam IndexTraitsT The index traits type.
 */
template <typename IndexTraitsT>
class StackedRGraph : public HierarchicalGraphV2<IndexTraitsT> {

    using base_t           = HierarchicalGraphV2<IndexTraitsT>;

    using vertex_num_t     = typename IndexTraitsT::vertex_num_t;
    using vertex_id_t      = typename IndexTraitsT::vertex_id_t;
    using layer_num_t      = typename IndexTraitsT::layer_num_t;
    using layer_id_t       = typename IndexTraitsT::layer_id_t;
    using distance_t       = typename IndexTraitsT::distance_t;
    using ratio_t          = typename IndexTraitsT::ratio_t;
    using vec_ele_t        = typename IndexTraitsT::vec_ele_t;
    using lnbr_t           = typename IndexTraitsT::lnbr_t;
    using vector_array_t   = typename IndexTraitsT::vector_array_t;
    using internal_graph_t = typename IndexTraitsT::internal_graph_t;

    static constexpr vertex_id_t invalid_vertex_id = IndexTraitsT::invalid_vertex_id;

    // ============================================================
    //  Queue traits adapter: extends IndexTraitsT with the extra
    //  type aliases that router data structures (CandidateEntry /
    //  StdCandidateQueue) require. Avoids touching the trait
    //  hierarchy itself — StackedRGraph is the only user.
    //
    //  Note: StdCandidateQueue's dist_func_t and visited_table_t are
    //  only referenced inside seeded_initialize / random_initialize,
    //  which we never call (we push entries manually). The stub
    //  types below satisfy the type-alias requirements without
    //  pulling in a ComputerTraits dependency.
    // ============================================================
    struct _queue_traits : public IndexTraitsT {
        struct _dist_func_stub {};
        using dist_func_t     = _dist_func_stub;
        using visited_table_t = typename IndexTraitsT::thread_local_bitmap_t;
        using candidate_entry_t = CandidateEntry<_queue_traits>;
        using knn_results_t     = std::vector<candidate_entry_t>;
        static constexpr candidate_entry_t invalid_candidate_entry =
            candidate_entry_t::make_invalid_entry();
        static constexpr candidate_entry_t min_candidate_entry =
            candidate_entry_t::make_min_entry();
    };

    using candidate_entry_t = typename _queue_traits::candidate_entry_t;
    using candidate_queue_t = StdCandidateQueue<_queue_traits>;
    using visited_table_t   = typename _queue_traits::visited_table_t;

public:
    /**
     * @brief Number of initial vertices inserted serially during bootstrap.
     *        While the hierarchy is still forming, concurrent insertions
     *        tend to create structural conflicts (too many threads trying
     *        to grow the same layer), so we serialize the first
     *        @c startup_points insertions before switching to TBB parallel.
     */
    static constexpr vertex_num_t startup_points = 5000;

    /** @brief Result type returned by candidate sets. */
    using candidate_t = std::pair<distance_t, lnbr_t>;

    /**
     * @brief Compute the recommended @c max_restrict_level for a dataset of
     *        @p total_vertices points as @c ceil(log_16(total_vertices / 1024)).
     *
     * Clamped to at least 1 so the hierarchy can always host one upper layer
     * even for very small datasets.
     */
    static auto compute_max_restrict_level(const vertex_num_t total_vertices)
        -> layer_num_t
    {
        if (total_vertices == 0) return 1;
        const double ratio = static_cast<double>(total_vertices) / 1024.0;
        if (ratio <= 1.0) return 1;
        const double raw = std::log(ratio) / std::log(16.0);
        const layer_num_t ceiled =
            static_cast<layer_num_t>(std::ceil(raw));
        return std::max<layer_num_t>(ceiled, layer_num_t(1));
    }

    /**
     * @brief Construct a StackedRGraph.
     *
     * @param rnet_beta            Radius growth factor: R_h = L1_rnet_radius * rnet_beta^(h-1).
     * @param L1_rnet_radius       Covering radius for layer 1 (the lowest upper layer).
     * @param max_restrict_level   Maximum layer (1-indexed) a single point may
     *                             reach. This is also the hard cap on the
     *                             total number of upper layers — the parent
     *                             @c HierarchicalGraphV2 pre-allocates exactly
     *                             this many layer slots. Use the static helper
     *                             @c compute_max_restrict_level(N) to derive it
     *                             from the dataset size as
     *                             @c ceil(log_16(N / 1024)).
     * @param search_nn_qs         Beam-search queue size used during Phase 1
     *                             top-down nearest-neighbor descent. A small
     *                             value (e.g. 40) is typically enough because
     *                             we only need the single closest vertex at
     *                             each layer.
     * @param select_nbrs_qs       Beam-search queue size used during Phase 2
     *                             candidate gathering before neighbor pruning.
     *                             A larger value (e.g. 500) gives the pruner
     *                             a richer candidate pool and better result
     *                             quality.
     * @param descent_fanout       Number of candidate vertices carried from one
     *                             layer to the next during Phase 1 top-down
     *                             descent. Each carried vertex becomes a
     *                             starting point for the next layer's beam
     *                             search, which dramatically reduces the
     *                             "wrong cluster" failure mode of single-entry
     *                             descent. Default 8.
     * @param descent_random_seeds Number of uniformly-random vertices to append
     *                             to the top layer's initial entry set (in
     *                             addition to the stored atomic entry point).
     *                             De-biases the very first beam search.
     *                             Default 3.
     * @param max_nbr_size         Per-vertex neighbor capacity for every layer.
     * @param layer_cap_ratio      Geometric decay base for per-layer capacity:
     *                             layer_h capacity = max(N / layer_cap_ratio^(h-1),
     *                             min_layer_cap). Must be @c >= 1. A value of
     *                             @c 1 (the default) means "no decay" — every
     *                             layer is sized to N, which is the safest
     *                             choice for datasets where many points remain
     *                             uncovered at the higher layers. Larger values
     *                             save memory but risk OOB if more than the
     *                             predicted number of points actually bubble up.
     * @param min_layer_cap        Floor on per-layer capacity (so tiny upper layers
     *                             still have room). Default 1024.
     */
    StackedRGraph(
        const ratio_t rnet_beta,
        const distance_t L1_rnet_radius,
        const layer_num_t max_restrict_level,
        const vertex_num_t search_nn_qs,
        const vertex_num_t select_nbrs_qs,
        const vertex_num_t descent_fanout = 8,
        const vertex_num_t descent_random_seeds = 3,
        const vertex_num_t max_nbr_size = 32,
        const ratio_t layer_cap_ratio = ratio_t(1),
        const vertex_num_t min_layer_cap = 1024
    ) :
        base_t(max_restrict_level),
        _rnet_beta(rnet_beta),
        _L1_rnet_radius(L1_rnet_radius),
        _max_restrict_level(max_restrict_level),
        _search_nn_qs(search_nn_qs),
        _select_nbrs_qs(select_nbrs_qs),
        _descent_fanout(descent_fanout),
        _descent_random_seeds(descent_random_seeds),
        _max_nbr_size(max_nbr_size),
        _layer_cap_ratio(layer_cap_ratio),
        _min_layer_cap(min_layer_cap)
    {
        if (rnet_beta <= ratio_t(1)) {
            ARTEA_ERROR(fmt::format("rnet_beta ({}) must be > 1", rnet_beta));
        }
        if (L1_rnet_radius <= distance_t(0)) {
            ARTEA_ERROR(fmt::format("L1_rnet_radius ({}) must be > 0", L1_rnet_radius));
        }
        if (max_restrict_level < 1) {
            ARTEA_ERROR(fmt::format("max_restrict_level ({}) must be >= 1",
                                    max_restrict_level));
        }
        if (search_nn_qs < 1) {
            ARTEA_ERROR(fmt::format("search_nn_qs ({}) must be >= 1", search_nn_qs));
        }
        if (select_nbrs_qs < 1) {
            ARTEA_ERROR(fmt::format("select_nbrs_qs ({}) must be >= 1", select_nbrs_qs));
        }
        if (descent_fanout < 1) {
            ARTEA_ERROR(fmt::format("descent_fanout ({}) must be >= 1", descent_fanout));
        }
        if (layer_cap_ratio < ratio_t(1)) {
            ARTEA_ERROR(fmt::format("layer_cap_ratio ({}) must be >= 1", layer_cap_ratio));
        }
    }

    // Inherits copy/move-deleted from HierarchicalGraphV2.

    // --- Tuning ---

    auto rnet_beta() const -> ratio_t { return _rnet_beta; }
    auto L1_rnet_radius() const -> distance_t { return _L1_rnet_radius; }
    auto max_nbr_size() const -> vertex_num_t { return _max_nbr_size; }
    auto max_restrict_level() const -> layer_num_t { return _max_restrict_level; }
    auto search_nn_qs() const -> vertex_num_t { return _search_nn_qs; }
    auto select_nbrs_qs() const -> vertex_num_t { return _select_nbrs_qs; }
    auto descent_fanout() const -> vertex_num_t { return _descent_fanout; }
    auto descent_random_seeds() const -> vertex_num_t { return _descent_random_seeds; }

    /**
     * @brief Covering radius for 1-indexed layer @p h (paper convention).
     *        @p h must be in [1, max_layers].
     */
    auto radius_at(const layer_id_t h) const -> distance_t {
        return static_cast<distance_t>(
            _L1_rnet_radius * std::pow(static_cast<double>(_rnet_beta),
                                       static_cast<double>(h - 1))
        );
    }

    /**
     * @brief Insert every vector in @p base_vecs as a new vertex.
     *
     * The first @c startup_points vectors are inserted serially; the rest
     * are inserted via @c tbb::parallel_for. The parallel phase calls the
     * same single-vertex primitive @c _insert_one as the serial phase, so
     * there is no code duplication.
     *
     * @tparam DistFuncT Callable: `(const vec_ele_t*, const vec_ele_t*) -> distance_t`.
     *         Must be thread-safe for concurrent reads.
     * @tparam SelectInitialFnT Callable selecting the initial neighbor set
     *         for the new vertex from sorted beam-search candidates.
     *         Signature:
     *         @code
     *         (const vec_ele_t* p_coords,
     *          distance_t R_h,
     *          vertex_num_t max_nbr_size,
     *          const std::vector<candidate_t>& sorted_candidates,
     *          const vector_array_t& base_vecs,
     *          DistFuncT& dist_func)
     *              -> std::vector<lnbr_t>
     *         @endcode
     *         May return any subset of the candidates (the implementation
     *         of ARC-Prune is delegated to the caller).
     * @tparam EvictFnT Callable passed to @c InternalGraph::add_nbr; invoked
     *         when a neighbor array is full to choose which neighbors to
     *         keep. Signature: `(lnbr_t* slots, lnbr_t new_nbr) -> uint64_t`.
     */
    template <typename DistFuncT, typename SelectInitialFnT, typename EvictFnT>
    auto add_vertices(
        const vector_array_t& base_vecs,
        DistFuncT&& dist_func,
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

        // Per-thread RNG for `_descent_random_seeds` top-layer sampling.
        // Seeded deterministically from the thread id so that parallel runs
        // produce reproducible (if thread-count-dependent) results.
        tbb::enumerable_thread_specific<std::mt19937> rng_pool(
            []() {
                static std::atomic<uint64_t> seed_ctr{0};
                return std::mt19937(static_cast<uint64_t>(0x9E3779B97F4A7C15ULL)
                                    ^ seed_ctr.fetch_add(1));
            });

        const vertex_num_t cutoff = std::min<vertex_num_t>(startup_points, n);

        // Serial startup. Uses the first thread-local bitmap/RNG slot; TBB
        // creates it lazily on first access from this thread.
        {
            auto& visited = visited_pool.local();
            auto& rng = rng_pool.local();
            for (vertex_num_t i = 0; i < cutoff; ++i) {
                _insert_one(i, base_vecs, dist_func, select_initial_fn,
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
                    _insert_one(i, base_vecs, dist_func, select_initial_fn,
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
     * Algorithm (matches the updated pseudocode in @c plans/starry-seeking-kahn.md):
     *
     *   Phase 1: Descend through EVERY layer from h_max down to 1, running
     *            a beam search at each and recording @c d_h* (the smallest
     *            candidate distance) plus the top-K candidates (for use as
     *            the next layer's entry set via inter_layer_link). No early
     *            stop.
     *
     *   Phase 1.5: Compute
     *              @c h_p = min{h ∈ [0, h_max - 1] : d_{h+1}* ≤ R_{h+1}}
     *                       ∪ {h_max + 1}
     *            That is, h_p is the SMALLEST h such that layer (h+1) covers
     *            p. Layers above the first covered one are deemed covered
     *            (even if technically some upper layer was uncovered — we
     *            only follow the lowest cover). If no layer covers p,
     *            h_p = h_max + 1, and we extend the hierarchy by one layer
     *            (capped by @c _max_restrict_level), seed the new top layer
     *            with p, and set h_p := h_max.
     *
     *   Phase 2: For h in [1, h_p], insert p at layer h unconditionally.
     *            Re-run beam search at each layer with the richer Phase 2
     *            queue size @c _select_nbrs_qs for the ARC-Prune candidate
     *            pool. Add reverse edges for each chosen neighbor.
     *
     *   Phase 2 epilogue: Patch each inserted vertex's inter_layer_link to
     *            point at p's layer_vid in the layer directly below. Also
     *            patch the Phase-1.5 extension vertex's link to point at p
     *            in layer h_max.
     *
     * @param visited Caller-owned bitmap; reused across insertions.
     * @param rng     Caller-owned RNG; reused across insertions for
     *                @c _descent_random_seeds top-layer sampling.
     */
    template <typename DistFuncT, typename SelectInitialFnT, typename EvictFnT,
              typename RngT>
    auto _insert_one(
        const vertex_id_t p,
        const vector_array_t& base_vecs,
        DistFuncT& dist_func,
        SelectInitialFnT& select_initial_fn,
        EvictFnT& evict_fn,
        visited_table_t& visited,
        RngT& rng
    ) -> void {
        const vec_ele_t* p_coords = base_vecs.get(p);

        // ---------- Phase 0: empty-hierarchy bootstrap ----------
        const layer_num_t h_max_before = this->get_num_layers();
        if (h_max_before == 0) {
            _ensure_layers_at_least(1, base_vecs);
            auto& layer0 = this->get_layer_graph(0);
            const vertex_id_t new_lv = layer0.add_vertex(p);
            this->update_entry_point(lnbr_t(p, new_lv));
            return;
        }

        const lnbr_t stored_ep = this->get_entry_point();
        if (stored_ep == lnbr_t::make_invalid_nbr()) {
            // Entry point not yet published. Fall back to bootstrapping at L1.
            _ensure_layers_at_least(1, base_vecs);
            auto& layer0 = this->get_layer_graph(0);
            const vertex_id_t new_lv = layer0.add_vertex(p);
            this->update_entry_point(lnbr_t(p, new_lv));
            return;
        }

        const layer_num_t h_max = this->get_num_layers();

        // ---------- Phase 1: beam-search descent at EVERY layer ----------
        //
        // No early stop. At each layer we run a small beam search of width
        // @c _search_nn_qs (= L_1) and cache the FULL candidate set C_h
        // (not just the top-K). Phase 2 later reuses C_h at the same layer
        // as the seed for its wider beam search (width _select_nbrs_qs),
        // which is the optimization in the updated pseudocode.
        //
        //   d_h_stars[h-1] = d_h* (smallest distance at layer h), or +∞ if
        //                    the layer's beam search returned empty.
        //   C_at[h-1]      = full (distance, lnbr_t) candidate set C_h
        //                    (up to _search_nn_qs entries, sorted ascending).
        std::vector<distance_t> d_h_stars(
            h_max, std::numeric_limits<distance_t>::max());
        std::vector<std::vector<candidate_t>> C_at(h_max);

        // Reusable buffer: the entry set for the NEXT beam search.
        std::vector<lnbr_t> current_entries;
        current_entries.reserve(
            static_cast<std::size_t>(_descent_fanout) +
            static_cast<std::size_t>(_descent_random_seeds) + 1);
        current_entries.push_back(stored_ep);

        // Seed the top layer with random vertices to de-bias the first
        // beam search (compensates for a stale / clustered stored_ep).
        if (_descent_random_seeds > 0) {
            const auto& top_layer = this->get_layer_graph(h_max - 1);
            const vertex_num_t top_size = top_layer.get_num_vertices();
            if (top_size > 1) {
                std::uniform_int_distribution<vertex_num_t>
                    uni_dist(0, top_size - 1);
                for (vertex_num_t i = 0; i < _descent_random_seeds; ++i) {
                    const vertex_id_t rand_lv = uni_dist(rng);
                    const vertex_id_t bv =
                        _resolve_base_vid(h_max - 1, rand_lv);
                    if (bv != invalid_vertex_id) {
                        current_entries.emplace_back(bv, rand_lv);
                    }
                }
            }
        }

        for (layer_num_t h = h_max; h >= 1; --h) {
            const layer_id_t layer_idx = h - 1;

            auto cands = _beam_search_layer(
                p_coords, base_vecs, dist_func, layer_idx,
                std::span<const lnbr_t>(current_entries),
                _search_nn_qs, visited);

            if (cands.empty()) {
                // Layer transiently empty (race) or all entries got clamped
                // out. d_h_stars[h-1] stays at +∞ (uncovered); C_at[h-1]
                // stays empty. Fall back to the stored entry point for the
                // next (lower) layer.
                if (h > 1) {
                    current_entries.clear();
                    current_entries.push_back(stored_ep);
                }
                continue;
            }

            d_h_stars[h - 1] = cands.front().first;

            // Translate top-K of `cands` to layer (h-1) entries via
            // inter_layer_link BEFORE moving `cands` into the cache.
            if (h > 1) {
                auto& layer_h = this->get_layer_graph(layer_idx);
                const std::size_t keep = std::min<std::size_t>(
                    _descent_fanout, cands.size());
                current_entries.clear();
                for (std::size_t i = 0; i < keep; ++i) {
                    const lnbr_t& c = cands[i].second;
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
        // Meaning: h_p is the SMALLEST h such that layer (h+1) already covers
        // p. Layers [h_p + 1 .. h_max] are deemed covered (even if a later
        // layer was uncovered — we stop at the first found cover). Layers
        // [1 .. h_p] are deemed uncovered and get an insertion in Phase 2.
        // If no layer covers p, h_p = h_max + 1 and we extend the hierarchy.
        layer_num_t h_p = h_max + 1;
        for (layer_num_t h = 0; h < h_max; ++h) {
            if (d_h_stars[h] <= radius_at(h + 1)) {
                h_p = h;
                break;
            }
        }

        // ---------- Phase 1.5: extend hierarchy if uncovered everywhere ----------
        //
        // Extension is capped by @c _max_restrict_level. If the cap is
        // reached we simply set h_p := h_max and proceed (p will still be
        // inserted at every existing layer, but no new layer is created).
        std::optional<std::pair<layer_id_t, vertex_id_t>> ext_slot;
        if (h_p == h_max + 1) {
            const layer_num_t new_h = h_max + 1;
            if (new_h > _max_restrict_level) {
                h_p = h_max;
            } else {
                const vertex_num_t base_n =
                    static_cast<vertex_num_t>(base_vecs.get_num_vecs());
                this->extend_and_seed(
                    new_h,
                    [&](const layer_id_t new_layer_id) {
                        const vertex_num_t cap =
                            _capacity_for_layer(new_layer_id, base_n);
                        return std::make_unique<internal_graph_t>(
                            cap, _max_nbr_size);
                    },
                    [&](const layer_id_t top_idx,
                        internal_graph_t& top_layer) -> lnbr_t {
                        // Placeholder inter_layer_link (= p). Patched by the
                        // Phase-2 epilogue once we know p's layer_vid in the
                        // layer directly below.
                        const vertex_id_t new_top_lv = top_layer.add_vertex(p);
                        ext_slot = std::make_pair(top_idx, new_top_lv);
                        return lnbr_t(p, new_top_lv);
                    });
                h_p = h_max;
            }
        }

        // ---------- Phase 2: insert p at layers [1, h_p] ----------
        //
        // By definition of h_p, every layer h in [1, h_p] has d_h* > R_h
        // according to Phase 1's beam search (i.e., "not absorbed at h").
        // We insert unconditionally at each such layer.
        if (h_p == 0) {
            // Covered at layer 1 already (and hence at every upper layer by
            // the "lowest cover" convention). No insertion in upper hierarchy.
            return;
        }

        // Reusable scratch buffer for converting a cached C_h into lnbr_t
        // entries for _beam_search_layer.
        std::vector<lnbr_t> phase2_entries;
        phase2_entries.reserve(static_cast<std::size_t>(_search_nn_qs) + 1);

        std::vector<std::pair<layer_id_t, vertex_id_t>> insertions;
        insertions.reserve(h_p);

        for (layer_num_t h = h_p; h >= 1; --h) {
            const layer_id_t layer_idx = h - 1;
            auto& layer_h = this->get_layer_graph(layer_idx);
            const distance_t R_h = radius_at(h);

            // Seed Phase 2's beam search from the cached Phase 1 candidate
            // set C_h at the SAME layer. This is the key optimization in
            // the updated pseudocode: we do not propagate phase2_entries
            // down via inter_layer_link between Phase 2 iterations; every
            // layer starts fresh from its own in-layer Phase 1 output.
            phase2_entries.clear();
            if (h - 1 < h_max && !C_at[h - 1].empty()) {
                for (const auto& [d, lv] : C_at[h - 1]) {
                    (void)d;
                    phase2_entries.push_back(lv);
                }
            } else {
                // C_h was empty (transient race in Phase 1). Fall back to
                // the stored entry point — _beam_search_layer will clamp it.
                phase2_entries.push_back(stored_ep);
            }

            // Rich beam search with _select_nbrs_qs (= L_2, e.g. 500) for
            // ARC-Prune.
            auto candidates = _beam_search_layer(
                p_coords, base_vecs, dist_func, layer_idx,
                std::span<const lnbr_t>(phase2_entries),
                _select_nbrs_qs, visited);

            // Insert p at this layer. Placeholder lower-layer vid is `p`
            // (the base_vid); upper layers will be patched in the epilogue.
            const vertex_id_t new_lv_h = layer_h.add_vertex(p);
            insertions.emplace_back(layer_idx, new_lv_h);
            const lnbr_t p_lnbr_h(p, new_lv_h);

            auto initial_nbrs = select_initial_fn(
                p_coords, R_h, _max_nbr_size, candidates, base_vecs, dist_func);

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
            auto& upper_layer = this->get_layer_graph(upper_idx);
            upper_layer.set_inter_layer_link(upper_lv, lower_lv);
        }

        // Patch the Phase-1.5 extension vertex's inter_layer_link to point
        // at p's layer_vid in layer h_max (the layer directly below the new
        // top). `insertions.front()` is the topmost Phase 2 layer (h = h_p
        // = h_max), so its new_lv is p's identity in that layer.
        if (ext_slot && !insertions.empty()) {
            auto& ext_layer = this->get_layer_graph(ext_slot->first);
            ext_layer.set_inter_layer_link(
                ext_slot->second, insertions.front().second);
        }
    }

    /**
     * @brief Walk the inter_layer_link chain from @p (layer_idx, layer_vid)
     *        all the way down to layer 0 and return the corresponding
     *        @c base_vid. Returns @c invalid_vertex_id if any step is OOB.
     */
    auto _resolve_base_vid(const layer_id_t layer_idx,
                           const vertex_id_t layer_vid) const -> vertex_id_t {
        vertex_id_t cur = layer_vid;
        for (layer_id_t down = layer_idx; down > 0; --down) {
            const auto& layer = this->get_layer_graph(down);
            if (cur >= layer.get_num_vertices()) return invalid_vertex_id;
            cur = layer.get_inter_layer_link(cur);
        }
        const auto& layer0 = this->get_layer_graph(0);
        if (cur >= layer0.get_num_vertices()) return invalid_vertex_id;
        return layer0.get_inter_layer_link(cur);
    }

    // ============================================================
    //  Layer extension helper
    // ============================================================

    /**
     * @brief Grow the parent hierarchy to at least @p target_num_layers layers.
     *        Each new layer is an @c InternalGraph sized by @c _capacity_for_layer.
     */
    auto _ensure_layers_at_least(
        const layer_num_t target_num_layers,
        const vector_array_t& base_vecs
    ) -> layer_num_t {
        const vertex_num_t base_n = static_cast<vertex_num_t>(base_vecs.get_num_vecs());
        return this->grow_layers(
            target_num_layers,
            [&](const layer_id_t new_layer_id) {
                const vertex_num_t cap = _capacity_for_layer(new_layer_id, base_n);
                return std::make_unique<internal_graph_t>(cap, _max_nbr_size);
            }
        );
    }

    /**
     * @brief Compute the per-layer CSR capacity (number of vertex slots) for
     *        the given 0-indexed layer id.
     *
     * Returns @c base_n when @c _layer_cap_ratio == 1 (no decay — every
     * layer pre-allocates for every base vertex, safe by construction but
     * memory-heavy). Otherwise returns
     * @c max(base_n / ratio^layer_id, min_layer_cap), which trades memory
     * for the assumption that upper layers hold geometrically fewer points.
     */
    auto _capacity_for_layer(const layer_id_t layer_id,
                             const vertex_num_t base_n) const -> vertex_num_t {
        if (_layer_cap_ratio == ratio_t(1)) {
            return base_n;
        }
        // Paper h = layer_id + 1; geometric decay starts at layer 1 (h=1) == N.
        const double divisor = std::pow(static_cast<double>(_layer_cap_ratio),
                                        static_cast<double>(layer_id));
        const double raw = static_cast<double>(base_n) / divisor;
        const vertex_num_t cap = static_cast<vertex_num_t>(raw);
        return std::max<vertex_num_t>(cap, _min_layer_cap);
    }

    // ============================================================
    //  Single-layer search helpers
    // ============================================================

    /**
     * @brief Beam search on a single layer starting from a set of @p entries.
     *        Returns up to @p queue_size (distance, lnbr_t) pairs sorted by
     *        ascending distance.
     *
     * Uses the router's @c StdCandidateQueue (via the internal
     * @c _queue_traits adapter) plus a caller-provided
     * @c ThreadLocalBitmap for visited tracking. The bitmap is reused
     * across calls (see @c add_vertices); the caller is responsible for
     * clearing it before the call.
     *
     * Clamps each entry's @c layer_vid against the current layer size and
     * clamps the raw neighbor count against @c max_nbr_size to defend
     * against stale/torn reads during concurrent extension.
     *
     * Reads are lock-free: the per-vertex spinlock is NOT acquired. Stale
     * reads of newly-inserted neighbors are tolerated (the algorithm is
     * designed to work under optimistic concurrency per the paper).
     *
     * @param visited Caller-owned visited bitmap. Must be sized to at least
     *                the current layer's vertex count. The function calls
     *                @c visited.clear() on entry.
     */
    template <typename DistFuncT>
    auto _beam_search_layer(
        const vec_ele_t* query_coords,
        const vector_array_t& base_vecs,
        DistFuncT& dist_func,
        const layer_id_t layer_idx,
        std::span<const lnbr_t> entries,
        const vertex_num_t queue_size,
        visited_table_t& visited
    ) const -> std::vector<candidate_t> {
        const auto& layer = this->get_layer_graph(layer_idx);
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

    // --- Members ---

    ratio_t _rnet_beta;
    distance_t _L1_rnet_radius;
    layer_num_t _max_restrict_level;
    vertex_num_t _search_nn_qs;
    vertex_num_t _select_nbrs_qs;
    vertex_num_t _descent_fanout;
    vertex_num_t _descent_random_seeds;
    vertex_num_t _max_nbr_size;
    ratio_t _layer_cap_ratio;
    vertex_num_t _min_layer_cap;
};

}   // namespace cpu
}   // namespace artea
