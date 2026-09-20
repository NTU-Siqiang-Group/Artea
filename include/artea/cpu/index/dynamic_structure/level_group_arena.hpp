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
 * @Description: Per-highest-level slot arena with pointer-stable blocks
 *               and a CAS bump counter fronted by thread-local chunks.
 */

#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <tbb/concurrent_vector.h>
#include <tbb/enumerable_thread_specific.h>

#include <artea/common/logger.hpp>
#include <artea/cpu/containers/allocator.hpp>

namespace artea {
namespace cpu {
namespace dynamic {

/**
 * @brief Per-@c highest_level_id slot pool used by @c HierarchicalGraph.
 *
 * Each block owns @c slots_per_block contiguous slots. A slot contains
 * @c slot_nbrs_count neighbors for one vertex, packed from its highest
 * level down to L0, and never crosses a block boundary. Offsets remain
 * logical @c nbr_t offsets; @c slot_ptr resolves them through the block
 * table instead of a single base pointer.
 *
 * Slot allocation uses a CAS-based bump counter fronted by a per-thread
 * slot-chunk cache (@c tbb::enumerable_thread_specific). Every thread,
 * on a cache miss, reserves @c slot_chunk_size slots at once via a single
 * successful CAS (or the final smaller batch at the capacity limit).
 * Subsequent claims use the thread-local cache without synchronization.
 *
 * Claims grow the arena on demand; callers may also reserve capacity with
 * @c ensure_slot_capacity. Growth is serialized, but cached claims and
 * existing neighbor access do not take the growth lock. Blocks are fully
 * constructed before a release-store publishes their capacity; claims
 * and lookup acquire that publication. Published
 * blocks are never resized or moved, so existing pointers stay valid.
 * The capacity publication does not synchronize writes to neighbors:
 * callers still own vertex initialization and neighbor locking.
 *
 * A data-block allocation failure leaves growth retryable. A block-table
 * append failure poisons further growth and propagates the exception;
 * the previously published prefix remains accessible. End the build and
 * join all users before destroying the arena. Destruction is not concurrent
 * with any operation.
 *
 * @tparam IndexTraitsT The index traits type (supplies nbr_t, vertex_num_t,
 *                      layer_id_t and the cache-aligned allocator policy).
 */
template <typename IndexTraitsT>
class LevelGroupArena {

    using nbr_t          = typename IndexTraitsT::nbr_t;
    using vertex_num_t   = typename IndexTraitsT::vertex_num_t;
    using layer_id_t     = typename IndexTraitsT::layer_id_t;

    using slot_block_t = cache_aligned_container_t<nbr_t>;

    static_assert(std::is_unsigned_v<vertex_num_t> && sizeof(vertex_num_t) <= sizeof(std::size_t));

    /** @brief Assumed L1-cache line size for anti-false-sharing padding. */
    static constexpr std::size_t cache_line_size = 64;

public:
    /** @brief Number of complete vertex slots in a normal data block. */
    static constexpr std::size_t slots_per_block = IndexTraitsT::slots_per_block;
    static_assert(slots_per_block > 0);

    /** @brief How many slots each thread acquires per successful CAS. */
    static constexpr vertex_num_t slot_chunk_size = 8;

    /**
     * @brief Per-thread slot-chunk cache.
     *
     * Stores the logical offset (in @c nbr_t units) of the
     * next slot to hand out — the same unit as
     * @c HierarchicalGraph::VertexInfo::slot_offset, so the claim path
     * does a single integer advance with no unit conversion.
     */
    struct ThreadLocalSlotChunk {
        /** @brief Offset of the next unconsumed slot (in @c nbr_t units). */
        std::size_t  next_slot_offset = 0;
        /** @brief Remaining unconsumed slots in the chunk. */
        vertex_num_t slots_remaining  = 0;
    };

    using tl_slot_chunk_t = ThreadLocalSlotChunk;

    /**
     * @brief Construct an empty arena bound to one @c highest_level_id.
     *
     * @param highest_level_id   The arena's group key. Stored only for
     *                           diagnostic logging.
     * @param slot_nbrs_count    How many @c nbr_t entries live in one
     *                           slot of this arena.
     */
    LevelGroupArena(const layer_id_t highest_level_id, const vertex_num_t slot_nbrs_count)
        : _highest_level_id(highest_level_id), _slot_nbrs_count(slot_nbrs_count) {
        if (slot_nbrs_count == 0) {
            throw std::invalid_argument("LevelGroupArena: slot length must be nonzero");
        }
        if (slot_nbrs_count > slot_block_t{}.max_size() / slots_per_block) {
            throw std::length_error("LevelGroupArena: block size exceeds vector limit");
        }
        _block_nbrs_count = slots_per_block * slot_nbrs_count;
        // Bound both logical offsets and reported data bytes.
        _max_slot_capacity = std::min<std::size_t>(
            std::numeric_limits<vertex_num_t>::max(),
            std::numeric_limits<std::size_t>::max() / sizeof(nbr_t) / slot_nbrs_count);
    }

    // Copy/move deleted: owns an atomic counter and a tbb TLS container.
    LevelGroupArena(const LevelGroupArena&)            = delete;
    LevelGroupArena& operator=(const LevelGroupArena&) = delete;
    LevelGroupArena(LevelGroupArena&&)                 = delete;
    LevelGroupArena& operator=(LevelGroupArena&&)      = delete;

    ~LevelGroupArena() = default;

    /**
     * @brief Publish enough complete blocks for @p target_slot_capacity.
     *        May run concurrently with growth, claims and neighbor access.
     *
     * The final block may be shorter at the representable capacity limit.
     * Publication is per block, so a failed request may still have grown
     * the valid prefix. Growth never changes the slot-claim counter.
     */
    auto ensure_slot_capacity(const vertex_num_t target_slot_capacity) -> void {
        if (target_slot_capacity <= slot_capacity()) return;

        if (target_slot_capacity > _max_slot_capacity) {
            throw std::length_error("LevelGroupArena: slot capacity overflows storage size");
        }

        std::lock_guard lock(_growth_mutex);
        std::size_t capacity = _slot_capacity.load(std::memory_order_relaxed);
        if (target_slot_capacity <= capacity) return;
        if (_growth_failure) std::rethrow_exception(_growth_failure);

        while (capacity < target_slot_capacity) {
            const std::size_t block_slots_count = std::min(slots_per_block, _max_slot_capacity - capacity);
            // Allocate locally: if this fails, the block table is untouched.
            slot_block_t slot_block(block_slots_count * _slot_nbrs_count);
            try {
                _slot_blocks.emplace_back(std::move(slot_block));
            } catch (...) {
                // A failed concurrent_vector allocation can break its tail.
                _growth_failure = std::current_exception();
                throw;
            }
            capacity += block_slots_count;
            _slot_capacity.store(static_cast<vertex_num_t>(capacity), std::memory_order_release);
        }
    }

    /**
     * @brief Claim exactly one slot. Fast path: integer advance on the
     *        thread-local cache. Slow path: ensure capacity, then CAS
     *        the bump counter to reserve another @c slot_chunk_size slots.
     *
     * @return Logical offset in @c nbr_t units, suitable for
     *         @c VertexInfo::slot_offset and @c slot_ptr.
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

    // ---- Slot access ----

    /**
     * @brief Resolve an aligned offset within the published capacity.
     *        The full slot is contiguous and stays valid across growth.
     *        Callers initialize their claimed slots before sharing them.
     */
    __attribute__((always_inline))
    auto slot_ptr(const std::size_t slot_offset) const -> const nbr_t* {
        const std::size_t capacity = slot_capacity();
        if (slot_offset >= capacity * _slot_nbrs_count || slot_offset % _slot_nbrs_count != 0) {
            throw std::out_of_range("LevelGroupArena: invalid or unpublished slot offset");
        }
        return _slot_blocks[slot_offset / _block_nbrs_count].data() + slot_offset % _block_nbrs_count;
    }

    __attribute__((always_inline))
    auto slot_ptr(const std::size_t slot_offset) -> nbr_t* {
        return const_cast<nbr_t*>(std::as_const(*this).slot_ptr(slot_offset));
    }

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
        return _slot_capacity.load(std::memory_order_acquire);
    }

    __attribute__((always_inline))
    auto num_claimed_slots() const -> std::size_t {
        return _next_slot_idx.load(std::memory_order_relaxed);
    }

    struct MemoryUsage {
        std::size_t data_bytes;
        std::size_t block_count;
        std::size_t block_table_capacity;
        std::size_t block_table_bytes;
    };

    /**
     * @brief Inspect only after growth has stopped. Data bytes count the
     *        published neighbor storage; table bytes estimate vector-object
     *        storage separately, excluding TBB/allocator bookkeeping.
     */
    auto memory_usage() const -> MemoryUsage {
        const std::size_t capacity = slot_capacity();
        const std::size_t table_capacity = _slot_blocks.capacity();
        return {
            capacity * _slot_nbrs_count * sizeof(nbr_t),
            capacity / slots_per_block + (capacity % slots_per_block != 0),
            table_capacity,
            table_capacity * sizeof(slot_block_t)
        };
    }

private:
    /**
     * @brief Reserve @c slot_chunk_size slots for @p local via CAS.
     *
     * Publish storage before advancing the counter. Allocation failure
     * leaves this claim and its TLS cache unchanged. A failed CAS refreshes
     * the high-water mark, so every retry checks capacity again.
     */
    auto _refill_local_chunk(tl_slot_chunk_t& local) -> void {
        std::size_t cur = _next_slot_idx.load(std::memory_order_relaxed);
        while (true) {
            if (cur >= _max_slot_capacity) {
                ARTEA_ERROR(fmt::format(
                    "LevelGroupArena[{}]: slot pool exhausted "
                    "(claimed={}, capacity={}, chunk_size={}).",
                    _highest_level_id, cur, _max_slot_capacity, slot_chunk_size));
            }
            const auto chunk_size = static_cast<vertex_num_t>(
                std::min<std::size_t>(slot_chunk_size, _max_slot_capacity - cur));
            const std::size_t next = cur + chunk_size;
            ensure_slot_capacity(static_cast<vertex_num_t>(next));
            if (_next_slot_idx.compare_exchange_weak(cur, next, std::memory_order_relaxed,
                                                    std::memory_order_relaxed)) {
                local.next_slot_offset = cur * _slot_nbrs_count;
                local.slots_remaining  = chunk_size;
                return;
            }
            // CAS failure already refreshed `cur`; loop and retry.
        }
    }

    // -----------------------------------------------------------------
    //   State
    // -----------------------------------------------------------------

    const layer_id_t   _highest_level_id;
    const vertex_num_t _slot_nbrs_count;   // nbr_t entries per slot
    std::size_t _block_nbrs_count = 0;
    std::size_t _max_slot_capacity = 0;

    /** @brief Append-only table of fixed-size, separately owned buffers. */
    tbb::concurrent_vector<slot_block_t> _slot_blocks;
    std::mutex _growth_mutex;
    std::exception_ptr _growth_failure;  // guarded by _growth_mutex

    /** @brief Constructed and published prefix, never the TBB table size. */
    std::atomic<vertex_num_t> _slot_capacity{0};

    /**
     * @brief Hot atomic bump counter (in slots). Isolated on its own
     *        cache line via @c alignas so it can't be in the same cache
     *        line as the read-only members above (which would cause the
     *        read-only members to ping-pong every time a thread bumps
     *        the counter).
     */
    alignas(cache_line_size) std::atomic<std::size_t> _next_slot_idx{0};

    /** @brief Per-thread slot-chunk cache fronting @c _next_slot_idx. */
    tbb::enumerable_thread_specific<tl_slot_chunk_t> _local_chunks;

};  // class LevelGroupArena

}   // namespace dynamic
}   // namespace cpu
}   // namespace artea
