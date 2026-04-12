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
 * @FilePath: /Artea/include/artea/cpu/index/dynamic_structure/level_group_arena.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Per-highest-level slot arena used by the new
 *               dynamic::HierarchicalGraph. Stores one contiguous
 *               pre-allocated buffer of nbr_t and hands out fixed-size
 *               slots to concurrent threads through a CAS-driven bump
 *               counter fronted by a thread-local slot-chunk cache.
 */

#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>

#include <tbb/enumerable_thread_specific.h>

#include <artea/common/logger.hpp>
#include <artea/cpu/containers/allocator.hpp>

namespace artea {
namespace cpu {
namespace dynamic {

/**
 * @brief Per-@c highest_level_id slot pool used by @c HierarchicalGraph.
 *
 * An arena owns one contiguous buffer of @c nbr_t whose logical unit is a
 * "slot" of @c slot_nbrs_count consecutive entries. Each vertex whose
 * highest_level_id matches this arena claims exactly one slot; that slot
 * stores the vertex's neighbors for all levels @c [0, highest_level_id]
 * packed from high to low.
 *
 * Slot allocation uses a CAS-based bump counter fronted by a per-thread
 * slot-chunk cache (@c tbb::enumerable_thread_specific). Every thread,
 * on a cache miss, reserves @c slot_chunk_size slots at once via a single
 * successful CAS; subsequent @c claim_slot calls are then served from the
 * thread-local cache with zero synchronization until it runs dry.
 *
 * The CAS pattern is "look-before-you-leap": we only attempt to advance
 * the counter if the target value still fits inside the current capacity.
 * We never post-fix a failed reservation with a subtraction, which would
 * otherwise let two threads overshoot capacity and both "roll back" to
 * the same index (handing out the same slot twice and silently
 * corrupting neighbor data).
 *
 * Growth policy (intentionally minimal for now)
 * ---------------------------------------------
 * Capacity is (re)sized only by @c ensure_slot_capacity, which is meant
 * to be called single-threadedly from
 * @c HierarchicalGraph::add_vertices while no thread is inside
 * @c claim_slot. Resizing is allowed only before any slot has been
 * claimed. If more slots are needed after claims have started the arena
 * raises @c ARTEA_ERROR — the @c N / 2^h decay heuristic applied by
 * @c HierarchicalGraph is expected to be generous enough in practice.
 *
 * @todo Once real workloads hit the static cap, reintroduce live growth
 *       so callers issued before growth can still resolve their
 *       @c slot_offset (offsets are stable across reallocations, which
 *       is the whole point of storing offsets rather than pointers).
 *
 * @tparam IndexTraitsT The index traits type (supplies nbr_t, vertex_num_t,
 *                      layer_id_t and the cache-aligned allocator policy).
 */
template <typename IndexTraitsT>
class LevelGroupArena {

    using nbr_t          = typename IndexTraitsT::nbr_t;
    using vertex_num_t   = typename IndexTraitsT::vertex_num_t;
    using layer_id_t     = typename IndexTraitsT::layer_id_t;

    using nbr_arena_container_t = cache_aligned_container_t<nbr_t>;

    /** @brief Assumed L1-cache line size for anti-false-sharing padding. */
    static constexpr std::size_t cache_line_size = 64;

public:
    /** @brief How many slots each thread acquires per successful CAS. */
    static constexpr vertex_num_t slot_chunk_size = 8;

    /**
     * @brief Per-thread slot-chunk cache.
     *
     * Stores the offset (in @c nbr_t units from the arena base) of the
     * next slot to hand out — the same unit as
     * @c HierarchicalGraph::VertexInfo::slot_offset, so the claim path
     * does a single integer advance with no unit conversion.
     */
    struct LocalSlotChunk {
        /** @brief Offset of the next unconsumed slot (in @c nbr_t units). */
        std::size_t  next_slot_offset = 0;
        /** @brief Remaining unconsumed slots in the chunk. */
        vertex_num_t slots_remaining  = 0;
    };

    /**
     * @brief Construct an empty arena bound to one @c highest_level_id.
     *
     * @param highest_level_id   The arena's group key. Stored only for
     *                           diagnostic logging.
     * @param slot_nbrs_count    How many @c nbr_t entries live in one
     *                           slot of this arena.
     */
    LevelGroupArena(
        const layer_id_t   highest_level_id,
        const vertex_num_t slot_nbrs_count
    ) :
        _highest_level_id(highest_level_id),
        _slot_nbrs_count(slot_nbrs_count),
        _slot_capacity(0),
        _next_slot_idx(0) {}

    // Copy/move deleted: owns an atomic counter and a tbb TLS container.
    LevelGroupArena(const LevelGroupArena&)            = delete;
    LevelGroupArena& operator=(const LevelGroupArena&) = delete;
    LevelGroupArena(LevelGroupArena&&)                 = delete;
    LevelGroupArena& operator=(LevelGroupArena&&)      = delete;

    ~LevelGroupArena() = default;

    /**
     * @brief Ensure the arena can host at least @p target_slot_capacity
     *        slots. Single-threaded — call from
     *        @c HierarchicalGraph::add_vertices before any concurrent
     *        @c claim_slot is in flight.
     *
     * Resizing is permitted only while no slot has been claimed yet.
     * Attempting to grow after claims have started raises
     * @c ARTEA_ERROR — growing would relocate @c _storage and while
     * offsets would stay valid, live threads could already be reading
     * via stale base pointers cached in registers.
     */
    auto ensure_slot_capacity(const vertex_num_t target_slot_capacity) -> void {
        if (target_slot_capacity <= _slot_capacity) return;

        if (_next_slot_idx.load(std::memory_order_relaxed) > 0) {
            // TODO: re-enable live growth. For now we rely on the
            // caller's N / 2^h decay heuristic being generous enough.
            ARTEA_ERROR(fmt::format(
                "LevelGroupArena[{}]: cannot grow from {} to {} slots — "
                "{} slots have already been claimed. TODO: implement "
                "pointer-stable growth.",
                _highest_level_id, _slot_capacity, target_slot_capacity,
                _next_slot_idx.load(std::memory_order_relaxed)));
        }

        _storage.resize(
            static_cast<std::size_t>(target_slot_capacity) * _slot_nbrs_count);
        _slot_capacity = target_slot_capacity;
    }

    /**
     * @brief Claim exactly one slot. Fast path: integer advance on the
     *        thread-local cache. Slow path: CAS the bump counter to
     *        reserve another @c slot_chunk_size slots.
     *
     * @return Offset (in @c nbr_t units) of the claimed slot, measured
     *         from the arena's @c base_ptr(). Suitable for persistent
     *         storage in @c VertexInfo::slot_offset.
     */
    auto claim_slot() -> std::size_t {
        auto& local = _local_chunks.local();
        if (local.slots_remaining == 0) {
            _refill_local_chunk(local);
        }
        const std::size_t offset = local.next_slot_offset;
        local.next_slot_offset += _slot_nbrs_count;
        --local.slots_remaining;
        return offset;
    }

    // ---- Base-pointer accessor ----

    /**
     * @brief Raw base pointer of the arena buffer. Combine with a
     *        @c slot_offset returned by @c claim_slot to resolve a slot.
     *        Reads straight from @c _storage.data() — no cached copy —
     *        so any future reallocation is automatically visible.
     */
    __attribute__((always_inline))
    auto base_ptr() const -> const nbr_t* { return _storage.data(); }

    __attribute__((always_inline))
    auto base_ptr() -> nbr_t* { return _storage.data(); }

    // ---- Diagnostics ----

    __attribute__((always_inline))
    auto highest_level_id() const -> layer_id_t {
        return _highest_level_id;
    }

    __attribute__((always_inline))
    auto slot_nbrs_count() const -> vertex_num_t {
        return _slot_nbrs_count;
    }

    __attribute__((always_inline))
    auto slot_capacity() const -> vertex_num_t {
        return _slot_capacity;
    }

    __attribute__((always_inline))
    auto num_claimed_slots() const -> std::size_t {
        return _next_slot_idx.load(std::memory_order_relaxed);
    }

private:
    /**
     * @brief Reserve @c slot_chunk_size slots for @p local via CAS.
     *
     * Look-before-you-leap: only attempt to advance the counter if the
     * target still fits inside @c _slot_capacity. If capacity is
     * exhausted raise @c ARTEA_ERROR without ever touching the counter.
     * This protects against the "overshoot then retract" pseudo-rollback
     * that would let the same slot be handed out twice.
     */
    auto _refill_local_chunk(LocalSlotChunk& local) -> void {
        std::size_t cur = _next_slot_idx.load(std::memory_order_relaxed);
        while (true) {
            const std::size_t next = cur + slot_chunk_size;
            if (next > _slot_capacity) {
                // TODO: reintroduce dynamic growth. See class-level note.
                ARTEA_ERROR(fmt::format(
                    "LevelGroupArena[{}]: slot pool exhausted "
                    "(claimed={}, capacity={}, chunk_size={}).",
                    _highest_level_id, cur, _slot_capacity, slot_chunk_size));
            }
            if (_next_slot_idx.compare_exchange_weak(
                    cur, next,
                    std::memory_order_relaxed,
                    std::memory_order_relaxed))
            {
                local.next_slot_offset = cur * _slot_nbrs_count;
                local.slots_remaining  = slot_chunk_size;
                return;
            }
            // CAS failure already refreshed `cur`; loop and retry.
        }
    }

    // -----------------------------------------------------------------
    //   State
    // -----------------------------------------------------------------

    layer_id_t   _highest_level_id;
    vertex_num_t _slot_nbrs_count;   // nbr_t entries per slot

    /** @brief Single contiguous buffer of this arena. Never reallocates
     *         once @c claim_slot has started — see @c ensure_slot_capacity. */
    nbr_arena_container_t _storage;

    /** @brief Current bump-allocator capacity (in slots). */
    vertex_num_t _slot_capacity;

    /**
     * @brief Hot atomic bump counter (in slots). Isolated on its own
     *        cache line via @c alignas so it can't be in the same cache
     *        line as the read-only members above (which would cause the
     *        read-only members to ping-pong every time a thread bumps
     *        the counter).
     */
    alignas(cache_line_size) std::atomic<std::size_t> _next_slot_idx;

    /** @brief Per-thread slot-chunk cache fronting @c _next_slot_idx. */
    tbb::enumerable_thread_specific<LocalSlotChunk> _local_chunks;

};  // class LevelGroupArena

}   // namespace dynamic
}   // namespace cpu
}   // namespace artea
