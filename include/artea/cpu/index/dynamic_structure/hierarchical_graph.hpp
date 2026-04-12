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
#include <limits>
#include <memory>
#include <span>
#include <vector>

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

    static constexpr vertex_id_t invalid_vertex_id = IndexTraitsT::invalid_vertex_id;

    /** @brief Sentinel @c highest_level_id for rows that have not yet
     *         been passed through @c assign_layer. Chosen to be
     *         distinguishable from any valid level id in the usable
     *         range @c [0, max_highest_level_id]. */
    static constexpr layer_id_t unassigned_highest_level_id =
        std::numeric_limits<layer_id_t>::max();

public:
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
     * @brief Construct an empty hierarchical graph.
     *
     * @param max_highest_level_id  Inclusive upper bound on the value of
     *                              @c highest_level_id for any vertex in
     *                              this graph. Determines how many
     *                              per-level-group arenas are pre-allocated
     *                              (one arena per possible
     *                              @c highest_level_id in
     *                              @c [0, max_highest_level_id]).
     * @param max_nbr_size          Per-vertex neighbor capacity at every
     *                              upper level. The bottom level
     *                              automatically uses @c 2 * max_nbr_size.
     */
    HierarchicalGraph(
        const layer_num_t  max_highest_level_id,
        const vertex_num_t max_nbr_size
    ) :
        _max_highest_level_id(max_highest_level_id),
        _max_nbr_size(max_nbr_size)
    {
        const std::size_t num_arenas =
            static_cast<std::size_t>(max_highest_level_id) + 1;
        _arenas.reserve(num_arenas);
        for (std::size_t h = 0; h < num_arenas; ++h) {
            _arenas.emplace_back(
                std::make_unique<level_group_arena_t>(
                    /*highest_level_id=*/static_cast<layer_id_t>(h),
                    /*slot_nbrs_count=*/_compute_slots_nbr_count(
                        static_cast<layer_id_t>(h))));
        }
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

        const vertex_num_t total_vertices =
            static_cast<vertex_num_t>(_vertex_info_table.size());

        // Grow every arena to its per-level-group decayed capacity based
        // on the freshly-updated total vertex count.
        for (std::size_t h = 0; h < _arenas.size(); ++h) {
            const vertex_num_t target_slot_capacity =
                _default_slot_capacity_for_arena(
                    static_cast<layer_id_t>(h), total_vertices);
            _arenas[h]->ensure_slot_capacity(target_slot_capacity);
        }

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
        if (highest_level_id > _max_highest_level_id) {
            ARTEA_ERROR(fmt::format(
                "assign_layer: highest_level_id ({}) exceeds "
                "max_highest_level_id ({})",
                highest_level_id, _max_highest_level_id));
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
    auto max_highest_level_id() const -> layer_id_t {
        return _max_highest_level_id;
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

private:
    // -----------------------------------------------------------------
    //   Helpers
    // -----------------------------------------------------------------

    /** @brief Slot size in @c nbr_t for a vertex with highest_level_id H.
     *         Upper levels 1..H each take @c max_nbr_size entries; level 0
     *         takes @c 2 * max_nbr_size. Total = (H + 2) * max_nbr_size. */
    __attribute__((always_inline))
    auto _compute_slots_nbr_count(const layer_id_t highest_level_id) const
        -> vertex_num_t
    {
        return static_cast<vertex_num_t>(highest_level_id + 2) * _max_nbr_size;
    }

    /** @brief Heuristic pre-allocation: arena @p highest_level_id expects
     *         to host roughly @c total_vertices / 2^highest_level_id
     *         vertices. Floored at @c min_arena_slot_capacity. */
    auto _default_slot_capacity_for_arena(
        const layer_id_t   highest_level_id,
        const vertex_num_t total_vertices
    ) const -> vertex_num_t {
        constexpr vertex_num_t min_arena_slot_capacity = 64;
        std::size_t cap = static_cast<std::size_t>(total_vertices);
        cap >>= highest_level_id;
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
            #if defined(__x86_64__) || defined(__i386__)
            __builtin_ia32_pause();
            #endif
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
    layer_num_t _max_highest_level_id;

    /** @brief Per-vertex neighbor capacity at every upper layer. The
     *         bottom layer implicitly uses @c 2 * _max_nbr_size. */
    vertex_num_t _max_nbr_size;

    /** @brief One arena per possible @c highest_level_id in
     *         @c [0, _max_highest_level_id]. */
    std::vector<std::unique_ptr<level_group_arena_t>> _arenas;

    /** @brief Per-vertex info records, indexed by @c vertex_id_t.
     *         @c std::deque never relocates existing elements on
     *         @c emplace_back, which is essential because @c VertexInfo
     *         is non-movable (it holds a @c std::atomic<uint8_t> lock). */
    std::deque<VertexInfo> _vertex_info_table;

};  // class HierarchicalGraph

}   // namespace dynamic
}   // namespace cpu
}   // namespace artea
