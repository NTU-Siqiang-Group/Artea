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
 * @FilePath: /Artea/include/artea/cpu/index/dynamic_structure/hierarchical_graph.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Dynamic hierarchical graph with a vertex-info table and
 *               one contiguous per-level-group neighbor arena each. A
 *               vertex whose highest_level_id == H owns one slot in the
 *               arena keyed by H; that slot stores the vertex's neighbors
 *               for every level H, H-1, ..., 1, 0 contiguously (high level
 *               first, bottom level last). The bottom level (level 0) gets
 *               twice the per-level neighbor capacity of upper levels.
 */

#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <execution>
#include <limits>
#include <memory>
#include <span>
#include <vector>

#include <immintrin.h>

#include <tbb/concurrent_vector.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include <artea/common/logger.hpp>
#include <artea/cpu/index/dynamic_structure/level_group_arena.hpp>

namespace artea {
namespace cpu {
namespace dynamic {

/**
 * @brief Dynamic hierarchical graph storing neighbors in per-level-group
 *        arenas with per-vertex spinlocks.
 *
 * Storage model
 * -------------
 * Every vertex is a row in @c _vertex_info_table whose entry carries:
 *   - @c nbrs_lock         — 1-byte spinlock protecting the vertex's
 *                            neighbor arrays against concurrent mutation.
 *   - @c highest_level_id  — the largest level at which this vertex
 *                            participates. Set by @c assign_layer().
 *   - @c slot_offset       — offset (in @c nbr_t units) of this vertex's
 *                            slot inside the arena keyed by
 *                            @c highest_level_id. Stored as an offset
 *                            rather than a raw pointer so it stays valid
 *                            across any future arena reallocation. Set
 *                            by @c assign_layer().
 *
 * Vertices that share the same @c highest_level_id share a single
 * @c LevelGroupArena. A vertex with @c highest_level_id == H owns one
 * slot of @c (H + 2) * max_nbr_size entries inside arena @c H. Within
 * that slot, neighbors are laid out contiguously from the highest level
 * down to the bottom level:
 *
 *   [ level H nbrs  | level H-1 nbrs | ... | level 1 nbrs | level 0 nbrs ]
 *      max_nbr_size    max_nbr_size          max_nbr_size   2*max_nbr_size
 *
 * Only the bottom level (level 0) gets the doubled per-vertex capacity.
 * Upper levels always use exactly @c max_nbr_size entries per vertex.
 *
 * @c fetch_layer_nbrs() composes the arena base pointer with the
 * vertex's @c slot_offset and the level-local offset with no branching:
 *   - Level offset: @c (H - level_id) * max_nbr_size  (correct for
 *     @c level_id == 0 because @c H - 0 == H, which places us at the
 *     start of the L0 block.)
 *   - Level length: @c max_nbr_size + max_nbr_size * (level_id == 0).
 *
 * Arena pre-allocation
 * --------------------
 * Arena @c h is pre-sized to hold roughly @c N / 2^h vertices where @c N
 * is the total vertex count after the latest @c add_vertices. Subsequent
 * @c add_vertices calls that would require a larger arena fail loudly —
 * see @c LevelGroupArena::ensure_slot_capacity for the policy. TODO:
 * reintroduce live growth once real workloads hit the static cap.
 *
 * Two-phase vertex creation
 * -------------------------
 *   1. @c add_vertices(N)       — reserves @c N contiguous vertex ids by
 *                                 appending @c N rows to
 *                                 @c _vertex_info_table. Every new row
 *                                 is left with a sentinel
 *                                 @c highest_level_id and a zero
 *                                 @c slot_offset; only the spinlock is
 *                                 usable until @c assign_layer runs.
 *   2. @c assign_layer(vid, H)  — decides the final @c highest_level_id
 *                                 and claims the slot inside arena @c H.
 *                                 After this call @c fetch_layer_nbrs()
 *                                 is callable.
 *
 * @tparam IndexTraitsT The index traits type.
 */
template <typename IndexTraitsT>
class HierarchicalGraph {

    using vertex_num_t = typename IndexTraitsT::vertex_num_t;
    using vertex_id_t  = typename IndexTraitsT::vertex_id_t;
    using layer_num_t  = typename IndexTraitsT::layer_num_t;
    using layer_id_t   = typename IndexTraitsT::layer_id_t;
    using nbr_t        = typename IndexTraitsT::nbr_t;

    using level_group_arena_t = LevelGroupArena<IndexTraitsT>;

public:
    static constexpr vertex_id_t invalid_vertex_id = IndexTraitsT::invalid_vertex_id;

    /** @brief Sentinel @c highest_level_id for rows that have not yet
     *         been passed through @c assign_layer. Chosen to be
     *         distinguishable from any valid level id in the usable
     *         range @c [0, max_restrict_level]. */
    static constexpr layer_id_t unassigned_highest_level_id =
        std::numeric_limits<layer_id_t>::max();

    /**
     * @brief Per-vertex row in @c _vertex_info_table.
     *
     * @c alignas(16) keeps the packed (lock, padding, highest_level_id)
     * plus @c slot_offset in one cache sub-line. The spinlock's
     * @c std::atomic is intentionally non-copy/move so the enclosing
     * record is non-movable — @c _vertex_info_table is therefore a
     * @c std::deque, which never relocates existing elements on
     * @c push_back.
     */
    struct alignas(16) VertexInfo {
        /** @brief 1-byte spinlock: 0 = unlocked, 1 = locked. */
        std::atomic<uint8_t> nbrs_lock;

        /** @brief Explicit padding keeping @c highest_level_id aligned. */
        uint8_t _padding[3];

        /** @brief Largest level at which this vertex participates. Valid
         *         only after @c assign_layer; until then it holds
         *         @c unassigned_highest_level_id. */
        layer_id_t highest_level_id;

        /** @brief Offset (in @c nbr_t units) of this vertex's slot
         *         inside the arena keyed by @c highest_level_id. Zero
         *         until @c assign_layer runs. */
        std::size_t slot_offset;

        VertexInfo()
            : nbrs_lock(0),
              _padding{0, 0, 0},
              highest_level_id(unassigned_highest_level_id),
              slot_offset(0) {}

        VertexInfo(const VertexInfo&)            = delete;
        VertexInfo& operator=(const VertexInfo&) = delete;
        VertexInfo(VertexInfo&&)                 = delete;
        VertexInfo& operator=(VertexInfo&&)      = delete;
    };

    /**
     * @brief Construct an empty hierarchical graph with every arena
     *        pre-sized so no later call to @c add_vertices needs to
     *        grow storage.
     *
     * @param max_restrict_level  Inclusive upper bound on the value of
     *                              @c highest_level_id for any vertex in
     *                              this graph. Determines how many
     *                              per-level-group arenas are allocated
     *                              (one arena per possible
     *                              @c highest_level_id in
     *                              @c [0, max_restrict_level]).
     * @param max_nbr_size          Per-vertex neighbor capacity at every
     *                              upper level. The bottom level
     *                              automatically uses @c 2 * max_nbr_size.
     * @param total_vertices        Expected eventual base-set size. Each
     *                              arena is sized to its worst-case cap
     *                              so live slot claims never need a
     *                              concurrent-unsafe @c resize.
     */
    HierarchicalGraph(
        const layer_num_t  max_restrict_level,
        const vertex_num_t max_nbr_size,
        const vertex_num_t total_vertices
    ) :
        _max_restrict_level(max_restrict_level),
        _max_nbr_size(max_nbr_size)
    {
        const std::size_t num_arenas =
            static_cast<std::size_t>(max_restrict_level) + 1;
        _arenas.reserve(num_arenas);
        for (std::size_t h = 0; h < num_arenas; ++h) {
            const layer_id_t layer = static_cast<layer_id_t>(h);
            auto arena = std::make_unique<level_group_arena_t>(
                /*highest_level_id=*/layer,
                /*slot_nbrs_count=*/_compute_slots_nbr_count(layer));
            arena->ensure_slot_capacity(
                _default_slot_capacity_for_arena(layer, total_vertices));
            _arenas.emplace_back(std::move(arena));
        }
        _vids_by_highest_level.resize(num_arenas);
    }

    // Copy and move both deleted: contains std::deque<VertexInfo> (whose
    // element is non-movable), std::unique_ptr<LevelGroupArena> members,
    // and the arenas themselves hold non-movable atomics + TLS state.
    HierarchicalGraph(const HierarchicalGraph&)            = delete;
    HierarchicalGraph& operator=(const HierarchicalGraph&) = delete;
    HierarchicalGraph(HierarchicalGraph&&)                 = delete;
    HierarchicalGraph& operator=(HierarchicalGraph&&)      = delete;

    ~HierarchicalGraph() = default;

    // =================================================================
    //   Phase 1: vertex-id reservation
    // =================================================================

    /**
     * @brief Append @p num_new_vertices rows to @c _vertex_info_table and
     *        grow every arena to match the new total vertex count.
     *
     * Thread-safety: NOT concurrent with itself or with @c assign_layer.
     * Callers invoke @c add_vertices once per batch from a single thread,
     * then hand the resulting vertex-id range to parallel
     * @c assign_layer workers.
     *
     * @param num_new_vertices  Number of rows to append.
     * @return The first newly-assigned vertex id (subsequent ids are
     *         contiguous up to @c first_id + num_new_vertices).
     */
    auto add_vertices(const vertex_num_t num_new_vertices) -> vertex_id_t {
        const vertex_id_t first_new_vertex_id =
            static_cast<vertex_id_t>(_vertex_info_table.size());
        if (num_new_vertices == 0) return first_new_vertex_id;

        for (vertex_num_t i = 0; i < num_new_vertices; ++i) {
            _vertex_info_table.emplace_back();
        }

        // Arenas are pre-sized at construction for the expected
        // total_vertices, so no resize-on-growth path is triggered here.
        return first_new_vertex_id;
    }

    // =================================================================
    //   Phase 2: per-vertex layer assignment
    // =================================================================

    /**
     * @brief Claim a neighbor-slot for vertex @p vid inside the arena
     *        indexed by @p highest_level_id and write the resulting
     *        offset + highest_level_id into @c _vertex_info_table[vid].
     *
     * Thread-safe: every thread services slot requests from its own
     * per-arena thread-local chunk (see @c LevelGroupArena::claim_slot).
     * Bulk-initializes the entire claimed slot to the invalid sentinel
     * so scans for the first invalid entry start in a clean state.
     */
    auto assign_layer(
        const vertex_id_t vid,
        const layer_id_t  highest_level_id
    ) -> void {
        if (highest_level_id > _max_restrict_level) {
            ARTEA_ERROR(fmt::format(
                "assign_layer: highest_level_id ({}) exceeds "
                "max_restrict_level ({})",
                highest_level_id, _max_restrict_level));
        }

        auto& arena = *_arenas[highest_level_id];
        const std::size_t slot_offset = arena.claim_slot();
        nbr_t* slot_base = arena.base_ptr() + slot_offset;

        // Bulk fill — nbr_t is trivially copyable (asserted in
        // neighbor.hpp) so std::fill_n lowers to a memset-style
        // unrolled copy on mainstream compilers/ISAs.
        std::fill_n(
            slot_base,
            _compute_slots_nbr_count(highest_level_id),
            nbr_t::make_invalid_nbr());

        auto& vinfo = _vertex_info_table[vid];
        vinfo.highest_level_id = highest_level_id;
        vinfo.slot_offset      = slot_offset;

        // Publish: ensure the vinfo writes above are visible to any thread
        // that later observes `vid` in _vids_by_highest_level. Without
        // this fence, a concurrent sample_entries / descent may read vid
        // from the bucket but see vinfo.highest_level_id ==
        // unassigned_highest_level_id, then index _arenas[unassigned] in
        // fetch_layer_nbrs and segfault.
        std::atomic_thread_fence(std::memory_order_release);

        // Record vid into the bucket keyed by its highest_level_id so
        // that router seeding and compactor materialization can both
        // enumerate this group without linear-scanning the full info
        // table.
        _vids_by_highest_level[highest_level_id].push_back(vid);

        // ORDERING IS LOAD-BEARING: the bucket push above must precede
        // the _top_occupied_level_id bump below. Readers of
        // top_occupied_level_id() do
        //     top = _top_occupied_level_id.load();
        //     bucket = _vids_by_highest_level[top];
        //     ...sample from bucket
        // If we bumped the top first, a reader could observe the new
        // top but find the bucket still empty (the pusher not having
        // reached push_back yet), producing a spurious "empty top
        // bucket" error. By ordering push_back → CAS, every observer
        // of an elevated top is guaranteed to see at least the vid
        // that caused the bump already present in the bucket.
        //
        // Monotonic CAS fetch-max: only try to raise the cached top,
        // never lower it. Release on success pairs with acquire in
        // top_occupied_level_id()'s reader.
        //
        // unassigned_highest_level_id is layer_id_t::max() (the largest
        // possible integer), so a naive `highest_level_id > prev_top`
        // check would be FALSE when prev_top is still the sentinel —
        // the cached top would stay at sentinel forever and every
        // reader would see "empty hierarchy". Handle the sentinel
        // explicitly as "smaller than any valid level".
        layer_id_t prev_top =
            _top_occupied_level_id.load(std::memory_order_relaxed);
        while ((prev_top == unassigned_highest_level_id ||
                highest_level_id > prev_top) &&
               !_top_occupied_level_id.compare_exchange_weak(
                   prev_top, highest_level_id,
                   std::memory_order_release,
                   std::memory_order_relaxed))
        {
            // CAS failed with prev_top refreshed; loop if we still see
            // a sentinel or a smaller cached top, otherwise exit
            // (another pusher already published an equal-or-higher
            // valid top for us).
        }
    }

    // =================================================================
    //   Per-layer neighbor access
    // =================================================================

    /**
     * @brief Return a mutable span over the neighbor array of vertex
     *        @p vid at level @p level_id.
     *
     * Branchless: the level-local offset and length both fold into
     * single arithmetic expressions. Precondition: @p level_id must be
     * in @c [0, get_highest_level_id(vid)].
     */
    __attribute__((always_inline))
    auto fetch_layer_nbrs(
        const vertex_id_t vid,
        const layer_id_t  level_id
    ) -> std::span<nbr_t> {
        const auto& vinfo = _vertex_info_table[vid];
        const layer_id_t H = vinfo.highest_level_id;
        nbr_t* slot_base = _arenas[H]->base_ptr() + vinfo.slot_offset;

        const std::size_t level_offset =
            static_cast<std::size_t>(H - level_id) * _max_nbr_size;
        const std::size_t level_nbrs_count =
            static_cast<std::size_t>(_max_nbr_size) +
            static_cast<std::size_t>(_max_nbr_size) *
                static_cast<std::size_t>(level_id == 0);

        return std::span<nbr_t>(slot_base + level_offset, level_nbrs_count);
    }

    __attribute__((always_inline))
    auto fetch_layer_nbrs(
        const vertex_id_t vid,
        const layer_id_t  level_id
    ) const -> std::span<const nbr_t> {
        const auto& vinfo = _vertex_info_table[vid];
        const layer_id_t H = vinfo.highest_level_id;
        const nbr_t* slot_base = _arenas[H]->base_ptr() + vinfo.slot_offset;

        const std::size_t level_offset =
            static_cast<std::size_t>(H - level_id) * _max_nbr_size;
        const std::size_t level_nbrs_count =
            static_cast<std::size_t>(_max_nbr_size) +
            static_cast<std::size_t>(_max_nbr_size) *
                static_cast<std::size_t>(level_id == 0);

        return std::span<const nbr_t>(slot_base + level_offset, level_nbrs_count);
    }

    /**
     * @brief Number of currently-valid neighbors at @p level_id for
     *        vertex @p vid, computed by scanning forward for the first
     *        invalid sentinel. O(max_nbr_size(level_id)).
     */
    __attribute__((always_inline))
    auto num_valid_nbrs(
        const vertex_id_t vid,
        const layer_id_t  level_id
    ) const -> vertex_num_t {
        const auto nbrs = fetch_layer_nbrs(vid, level_id);
        vertex_num_t count = 0;
        for (const auto& nbr : nbrs) {
            if (nbr.is_invalid()) break;
            ++count;
        }
        return count;
    }

    // =================================================================
    //   Per-vertex spinlocked mutation
    // =================================================================

    /**
     * @brief Acquire vertex @p vid's spinlock, invoke @p fn on a mutable
     *        span of its level-@p level_id neighbors, release the lock.
     *
     * @p fn is invoked as
     *   @code fn(std::span<nbr_t> nbrs, vertex_num_t current_valid_count) @endcode
     * where @c current_valid_count is the index of the first invalid
     * entry on entry. The callable should leave the span with its valid
     * prefix still terminated by an invalid sentinel.
     */
    template <typename FnT>
    auto with_locked_nbrs(
        const vertex_id_t vid,
        const layer_id_t  level_id,
        FnT&&             fn
    ) -> void {
        auto& vinfo = _vertex_info_table[vid];
        _acquire_nbrs_lock(vinfo);

        auto nbrs = fetch_layer_nbrs(vid, level_id);
        vertex_num_t current_valid_count = 0;
        for (const auto& nbr : nbrs) {
            if (nbr.is_invalid()) break;
            ++current_valid_count;
        }

        fn(nbrs, current_valid_count);

        _release_nbrs_lock(vinfo);
    }

    // =================================================================
    //   Properties / accessors
    // =================================================================

    __attribute__((always_inline))
    auto max_restrict_level() const -> layer_id_t {
        return _max_restrict_level;
    }

    /** @brief Per-vertex neighbor capacity at @p level_id (double for L0). */
    __attribute__((always_inline))
    auto max_nbr_size(const layer_id_t level_id) const -> vertex_num_t {
        return _max_nbr_size +
               _max_nbr_size * static_cast<vertex_num_t>(level_id == 0);
    }

    /** @brief Per-vertex neighbor capacity at every upper layer
     *         (level 0 capacity is 2x this). */
    __attribute__((always_inline))
    auto max_nbr_size() const -> vertex_num_t {
        return _max_nbr_size;
    }

    __attribute__((always_inline))
    auto get_num_vertices() const -> vertex_num_t {
        return static_cast<vertex_num_t>(_vertex_info_table.size());
    }

    __attribute__((always_inline))
    auto get_highest_level_id(const vertex_id_t vid) const -> layer_id_t {
        return _vertex_info_table[vid].highest_level_id;
    }

    __attribute__((always_inline))
    auto is_vertex_assigned(const vertex_id_t vid) const -> bool {
        return _vertex_info_table[vid].highest_level_id !=
               unassigned_highest_level_id;
    }

    __attribute__((always_inline))
    auto get_vertex_info_table() const -> const std::deque<VertexInfo>& {
        return _vertex_info_table;
    }

    /**
     * @brief Vids whose @c highest_level_id equals @p h. Populated by
     *        @c assign_layer; consumed by router seeding and the
     *        compactor.
     */
    __attribute__((always_inline))
    auto get_vids_with_highest_level(const layer_id_t h) const
        -> const tbb::concurrent_vector<vertex_id_t>&
    {
        return _vids_by_highest_level[h];
    }

    /**
     * @brief Largest @c h in @c [0, max_restrict_level] with a non-empty
     *        bucket, or @c unassigned_highest_level_id if every bucket
     *        is empty (i.e. no vertex has been assigned yet).
     *
     * Single atomic load of @c _top_occupied_level_id, maintained by
     * @c assign_layer via a CAS fetch-max. The push_back → CAS ordering
     * inside @c assign_layer guarantees that any observer of the
     * returned top sees at least one vid in the corresponding bucket.
     */
    auto top_occupied_level_id() const -> layer_id_t {
        return _top_occupied_level_id.load(std::memory_order_acquire);
    }

    /** @brief Per-vertex slot offset inside its group's arena
     *         (nbr_t-count units from arena base). Valid only after
     *         @c assign_layer. */
    __attribute__((always_inline))
    auto get_slot_offset(const vertex_id_t vid) const -> std::size_t {
        return _vertex_info_table[vid].slot_offset;
    }

    /** @brief Current bump-allocator capacity (in slots) of arena @p h.
     *         Used by the compactor to size the compact arena. */
    __attribute__((always_inline))
    auto get_arena_capacity_in_arena(const layer_id_t h) const -> vertex_num_t {
        return _arenas[h]->slot_capacity();
    }

    // =================================================================
    //   Layer ↔ RefiningGraph conversion
    // =================================================================

    /**
     * @brief Build the (local_to_global, global_to_local) maps for the
     *        participating-vid set at @p level_id.
     *
     * For @p level_id == 0, both maps are returned empty (identity mode):
     *        the caller should construct the @c RefiningGraph with the
     *        dense ctor.
     *
     * For @p level_id >= 1, walks every bucket
     *        @c _vids_by_highest_level[h] for @c h in @c [level_id,
     *        max_restrict_level], concatenates the vids, sorts ascending
     *        for reproducible row order, and builds the inverse map of
     *        size @c get_num_vertices() with @c invalid_vertex_id in
     *        non-participating slots.
     */
    auto collect_layer_vids(const layer_id_t level_id) const
        -> std::pair<std::vector<vertex_id_t>, std::vector<vertex_id_t>>
    {
        if (level_id == 0) {
            return {std::vector<vertex_id_t>{},
                    std::vector<vertex_id_t>{}};
        }

        // Phase 1: prefix-sum bucket sizes so each bucket gets its own
        // disjoint write window in the result vector. Cheap and serial —
        // (max_restrict_level + 1) is typically ≤ 20.
        const std::size_t num_buckets =
            static_cast<std::size_t>(_max_restrict_level - level_id + 1);
        std::vector<std::size_t> offsets(num_buckets + 1, 0);
        for (std::size_t k = 0; k < num_buckets; ++k) {
            const layer_id_t h = static_cast<layer_id_t>(level_id + k);
            offsets[k + 1] = offsets[k] + _vids_by_highest_level[h].size();
        }
        std::vector<vertex_id_t> local_to_global(offsets.back());

        // Phase 2: parallel bulk-copy each bucket into its window.
        tbb::parallel_for(
            tbb::blocked_range<std::size_t>(0, num_buckets),
            [&](const tbb::blocked_range<std::size_t>& r) {
                for (std::size_t k = r.begin(); k != r.end(); ++k) {
                    const layer_id_t h = static_cast<layer_id_t>(level_id + k);
                    const auto& bucket = _vids_by_highest_level[h];
                    std::copy(bucket.begin(), bucket.end(),
                              local_to_global.begin() + offsets[k]);
                }
            });

        // Phase 3: parallel sort. Stable order is not required; just need
        // reproducible cache-friendly layout.
        std::sort(std::execution::par,
                  local_to_global.begin(), local_to_global.end());

        // Phase 4: build the inverse map. First fill with invalid in
        // parallel, then scatter local indices.
        const vertex_num_t n_global = get_num_vertices();
        std::vector<vertex_id_t> global_to_local(n_global);
        tbb::parallel_for(
            tbb::blocked_range<vertex_num_t>(0, n_global),
            [&](const tbb::blocked_range<vertex_num_t>& r) {
                std::fill(global_to_local.begin() + r.begin(),
                          global_to_local.begin() + r.end(),
                          invalid_vertex_id);
            });
        tbb::parallel_for(
            tbb::blocked_range<std::size_t>(0, local_to_global.size()),
            [&](const tbb::blocked_range<std::size_t>& r) {
                for (std::size_t i = r.begin(); i != r.end(); ++i) {
                    global_to_local[local_to_global[i]] =
                        static_cast<vertex_id_t>(i);
                }
            });
        return {std::move(local_to_global), std::move(global_to_local)};
    }

    /**
     * @brief Populate @p refining_graph's @c _nbrs_arr from level
     *        @p level_id of this hierarchical graph.
     *
     * The caller must already have constructed @p refining_graph with
     * the matching mapping (dense ctor for L0; sparse ctor with the
     * maps from @c collect_layer_vids for upper layers).
     *
     * Embarrassingly parallel: each iteration writes only its own row.
     * No locks needed — extraction is a phase-boundary operation.
     */
    template <typename RefiningGraphT>
    auto fill_refining_graph_from_layer(
        RefiningGraphT&  refining_graph,
        const layer_id_t level_id
    ) const -> void {
        // Take min(src_capacity, dest_capacity), so callers can drive the
        // refiner with an RG layer cap that differs from the hg slot cap
        // in either direction. Symmetric with
        // writeback_layer_from_refining_graph.
        const vertex_num_t src_capacity = max_nbr_size(level_id);
        const vertex_num_t dest_capacity = refining_graph.layer_config().max_nbr_size();
        const vertex_num_t copy_capacity = std::min(src_capacity, dest_capacity);
        refining_graph.parallel_for_each_vertex(
            [&](const vertex_id_t /*local_vid*/, const vertex_id_t global_vid) {
                // Skip vids that were never assign_layer'd. Possible when
                // the caller constructed a dense RG with a vector array
                // larger than the assigned set (e.g. reusing the global
                // dataset for a partial fixture). Not produced by the
                // standard add_vertices → assign_layer flow, so warn.
                if (!is_vertex_assigned(global_vid)) {
                    ARTEA_WARN(fmt::format(
                        "fill_refining_graph_from_layer: skipping "
                        "unassigned vid={} (level_id={})",
                        global_vid, level_id));
                    return;
                }
                const auto src = fetch_layer_nbrs(global_vid, level_id);
                auto& dst = refining_graph.fetch_nbrs(global_vid);
                dst.clear();
                for (vertex_num_t i = 0; i < copy_capacity && i < src.size(); ++i) {
                    if (src[i].is_invalid()) break;
                    dst.push_back(src[i]);
                }
            });
    }

    /**
     * @brief Write @p refining_graph's neighbor lists back into level
     *        @p level_id of this hierarchical graph.
     *
     * Per-vertex spinlock guards each write; in practice writeback runs
     * as a phase boundary so contention is nil. The destination slot is
     * truncated to its capacity for @p level_id (handles the L0 2× cap
     * automatically) and terminated with the invalid sentinel.
     */
    template <typename RefiningGraphT>
    auto writeback_layer_from_refining_graph(
        RefiningGraphT&  refining_graph,
        const layer_id_t level_id
    ) -> void {
        refining_graph.parallel_for_each_vertex(
            [&](const vertex_id_t /*local_vid*/, const vertex_id_t global_vid) {
                if (!is_vertex_assigned(global_vid)) {
                    ARTEA_WARN(fmt::format(
                        "writeback_layer_from_refining_graph: skipping "
                        "unassigned vid={} (level_id={})",
                        global_vid, level_id));
                    return;
                }
                const auto& src = refining_graph.fetch_nbrs(global_vid);
                with_locked_nbrs(global_vid, level_id,
                    [&](std::span<nbr_t> dst, vertex_num_t /*old_cnt*/) {
                        const std::size_t copy_n =
                            std::min(src.size(), dst.size());
                        for (std::size_t i = 0; i < copy_n; ++i) {
                            dst[i] = src[i];
                        }
                        if (copy_n < dst.size()) {
                            dst[copy_n] = nbr_t::make_invalid_nbr();
                        }
                    });
            });
    }

private:
    // -----------------------------------------------------------------
    //   Helpers
    // -----------------------------------------------------------------

    /** @brief Slot size in @c nbr_t for a vertex with highest_level_id H.
     *         Upper levels 1..H each take @c max_nbr_size entries; level 0
     *         takes @c 2 * max_nbr_size. Total = (H + 2) * max_nbr_size. */
    __attribute__((always_inline))
    auto _compute_slots_nbr_count(const layer_id_t highest_level_id) const -> vertex_num_t {
        return static_cast<vertex_num_t>(highest_level_id + 2) * _max_nbr_size;
    }

    /** @brief Heuristic pre-allocation: arena @p highest_level_id expects
     *         to host up to @c total_vertices / conservative_beta^h
     *         vertices. Using a pessimistic beta (1.2) + 2x safety
     *         absorbs early-build skew (sparse upper layers inflate
     *         NN estimates and push more vertices into arena[1..2] than
     *         the steady-state r-net geometry predicts). Hard-capped
     *         at @c total_vertices — any single arena trivially
     *         bounded by N. Floored at @c min_arena_slot_capacity. */
    auto _default_slot_capacity_for_arena(
        const layer_id_t   highest_level_id,
        const vertex_num_t total_vertices
    ) const -> vertex_num_t {
        constexpr vertex_num_t min_arena_slot_capacity = 64;
        constexpr double conservative_beta  = 1.2;
        constexpr double safety_multiplier  = 2.0;

        double cap = static_cast<double>(total_vertices) * safety_multiplier;
        for (layer_id_t i = 0; i < highest_level_id; ++i) {
            cap /= conservative_beta;
        }
        cap = std::min(cap, static_cast<double>(total_vertices));

        return std::max<vertex_num_t>(
            static_cast<vertex_num_t>(cap), min_arena_slot_capacity);
    }

    __attribute__((always_inline))
    auto _acquire_nbrs_lock(VertexInfo& vinfo) const -> void {
        uint8_t expected = 0;
        while (!vinfo.nbrs_lock.compare_exchange_weak(
                   expected, 1,
                   std::memory_order_acquire,
                   std::memory_order_relaxed)) {
            expected = 0;
            _mm_pause();
        }
    }

    __attribute__((always_inline))
    auto _release_nbrs_lock(VertexInfo& vinfo) const -> void {
        vinfo.nbrs_lock.store(0, std::memory_order_release);
    }

    // -----------------------------------------------------------------
    //   State
    // -----------------------------------------------------------------

    /** @brief Inclusive upper bound of @c highest_level_id for any vertex. */
    layer_num_t _max_restrict_level;

    /** @brief Per-vertex neighbor capacity at every upper layer. The
     *         bottom layer implicitly uses @c 2 * _max_nbr_size. */
    vertex_num_t _max_nbr_size;

    /** @brief One arena per possible @c highest_level_id in
     *         @c [0, _max_restrict_level]. */
    std::vector<std::unique_ptr<level_group_arena_t>> _arenas;

    /** @brief Per-vertex info records, indexed by @c vertex_id_t.
     *         @c std::deque never relocates existing elements on
     *         @c emplace_back, which is essential because @c VertexInfo
     *         is non-movable (it holds a @c std::atomic<uint8_t> lock). */
    std::deque<VertexInfo> _vertex_info_table;

    /** @brief @c _vids_by_highest_level[h] holds every vid whose
     *         @c highest_level_id == h. Populated concurrently by
     *         @c assign_layer and consumed by
     *           (1) SingleLayerRouter::sample_entries — seeds top-level
     *               beam search with random entry vertices;
     *           (2) HierarchicalGraphCompactor — materializes the compact
     *               graph by walking each bucket in order. */
    std::vector<tbb::concurrent_vector<vertex_id_t>> _vids_by_highest_level;

    /** @brief Cached monotonically non-decreasing top-occupied level.
     *         Maintained by @c assign_layer via a CAS fetch-max AFTER the
     *         new vid has been pushed into its bucket, so any reader that
     *         observes an elevated top is guaranteed to also see at least
     *         that one vid in the corresponding bucket. Read by
     *         @c top_occupied_level_id() as a single atomic load instead
     *         of scanning every bucket. */
    std::atomic<layer_id_t> _top_occupied_level_id{unassigned_highest_level_id};

};  // class HierarchicalGraph

}   // namespace dynamic
}   // namespace cpu
}   // namespace artea
