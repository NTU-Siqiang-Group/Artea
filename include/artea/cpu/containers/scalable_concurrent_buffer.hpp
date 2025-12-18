// Copyright 2025 Weitang Ye
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

#if 0

#pragma once

#include <atomic>
#include <vector>
#include <memory>
#include <cassert>
#include <algorithm> // for std::max

namespace artea {
namespace cpu {

/**
 * @brief A scalable, thread-safe buffer that grows dynamically using a segmented architecture.
 *
 * It maintains the cache-line interleaving optimization of the fixed-size ConcurrentBuffer
 * within each segment to minimize false sharing.
 *
 * @tparam T Element type
 * @tparam block_capacity The capacity of each dynamically allocated block (segment).
 *                        Must be a multiple of (CACHE_LINE_SIZE / sizeof(T)).
 * @tparam max_total_capacity The theoretical hard limit for the buffer size (used to pre-allocate the directory spine).
 *                            Default is ~1 Billion elements if T is small.
 */
template <
    typename T,
    std::size_t block_capacity = 1048576, // Default block size: 1M elements
    std::size_t max_total_capacity = 1073741824 // Max 1B elements limit for the directory spine
>
struct alignas(CACHE_LINE_SIZE) ScalableConcurrentBuffer {

    static_assert(block_capacity > 0, "Block capacity must be positive");

    // Calculate directory size needed
    static constexpr std::size_t max_segments = (max_total_capacity + block_capacity - 1) / block_capacity;

    // =========================================================================
    // Inner Segment Class
    // =========================================================================
    struct alignas(CACHE_LINE_SIZE) Segment {
        static constexpr std::size_t num_elements_per_cacheline = CACHE_LINE_SIZE / sizeof(T);
        static constexpr std::size_t num_cachelines = block_capacity / num_elements_per_cacheline;
        static constexpr std::size_t cacheline_stride = num_elements_per_cacheline;

        // Raw storage for this segment
        std::vector<T> data;

        Segment() {
            data.resize(block_capacity);
        }

        // Wait-Free write at a specific logical offset within this segment
        inline void write_at(std::size_t logical_offset, const T& element) {
            // Apply the Interleaving Strategy
            const std::size_t row = logical_offset % num_cachelines;
            const std::size_t col = logical_offset / num_cachelines;
            const std::size_t physical_idx = (row * cacheline_stride) + col;

            data[physical_idx] = element;
        }

        // Read back resolving the interleaving
        inline T& read_at(std::size_t logical_offset) {
            const std::size_t row = logical_offset % num_cachelines;
            const std::size_t col = logical_offset / num_cachelines;
            const std::size_t physical_idx = (row * cacheline_stride) + col;

            return data[physical_idx];
        }

        // Move-read back
        inline T move_at(std::size_t logical_offset) {
            const std::size_t row = logical_offset % num_cachelines;
            const std::size_t col = logical_offset / num_cachelines;
            const std::size_t physical_idx = (row * cacheline_stride) + col;

            return std::move(data[physical_idx]);
        }
    };

    // =========================================================================
    // Members
    // =========================================================================

    /** @brief Global monotonic counter for logical indices */
    alignas(CACHE_LINE_SIZE) std::atomic<std::size_t> write_idx {0};

    /**
     * @brief Directory of pointers to segments.
     * We use an array of atomics to allow lock-free lazy initialization of segments.
     * Using a raw array/unique_ptr instead of vector to avoid resizing locks.
     */
    std::unique_ptr<std::atomic<Segment*>[]> segments;

    // =========================================================================
    // Methods
    // =========================================================================

    ScalableConcurrentBuffer() {
        // Allocate the spine (directory). This is small (e.g., 8KB for 1000 blocks).
        // We initialize all pointers to nullptr.
        segments = std::make_unique<std::atomic<Segment*>[]>(max_segments);
        for (std::size_t i = 0; i < max_segments; ++i) {
            segments[i].store(nullptr, std::memory_order_relaxed);
        }
    }

    ~ScalableConcurrentBuffer() {
        // Cleanup all allocated segments
        for (std::size_t i = 0; i < max_segments; ++i) {
            Segment* seg = segments[i].load(std::memory_order_relaxed);
            if (seg) {
                delete seg;
            }
        }
    }

    /**
     * @brief Thread-safe (Wait-Free mostly) append.
     * Only blocks briefly if a new segment needs to be allocated (rare).
     */
    auto append(const T& element) -> void {
        // 1. Reserve logical slot
        const std::size_t logical_idx = write_idx.fetch_add(1, std::memory_order_relaxed);

        if (logical_idx >= max_total_capacity) {
            // Hard limit reached. In production, handle this gracefully or abort.
            // For simplicity here, we assume max_total_capacity is sufficient.
            return;
        }

        // 2. Locate Segment and Offset
        const std::size_t seg_id = logical_idx / block_capacity;
        const std::size_t offset = logical_idx % block_capacity;

        // 3. Get Segment Pointer (Lazy Allocation)
        // Optimistic load: mostly it will be non-null.
        Segment* seg = segments[seg_id].load(std::memory_order_acquire);

        if (__builtin_expect(seg == nullptr, 0)) {
            // Create a new segment
            Segment* new_seg = new Segment();

            // Try to set it atomically (CAS)
            Segment* expected = nullptr;
            if (segments[seg_id].compare_exchange_strong(expected, new_seg,
                                                         std::memory_order_acq_rel,
                                                         std::memory_order_acquire)) {
                // We won the race, use the new segment
                seg = new_seg;
            } else {
                // We lost the race, someone else allocated it.
                // Delete our redundant object and use the one already there.
                delete new_seg;
                seg = expected;
            }
        }

        // 4. Write data using the optimized interleaving layout
        seg->write_at(offset, element);
    }

    /**
     * @brief Flushes all elements to a linear vector.
     * Not thread-safe (Caller must ensure no active writers).
     */
    auto flush() -> std::vector<T> {
        const std::size_t count = write_idx.load(std::memory_order_acquire);
        std::vector<T> result;
        result.reserve(count);

        for (std::size_t i = 0; i < count; ++i) {
            const std::size_t seg_id = i / block_capacity;
            const std::size_t offset = i % block_capacity;

            Segment* seg = segments[seg_id].load(std::memory_order_relaxed);

            // Should not happen if count is consistent, but safety check:
            if (seg) {
                result.push_back(seg->move_at(offset));
            }
        }

        // Reset: We keep the allocated segments for reuse to avoid malloc overhead
        // in the next iteration (common in Graph Iterations).
        write_idx.store(0, std::memory_order_release);

        return result;
    }

    /**
     * @brief Reset logic without deallocating memory blocks.
     */
    auto clear() -> void {
        write_idx.store(0, std::memory_order_relaxed);
    }

    auto size() const -> std::size_t {
        return write_idx.load(std::memory_order_relaxed);
    }

    /**
     * @brief Explicitly free memory if needed (e.g. at end of a major phase).
     */
    auto shrink_to_fit() -> void {
        // Reset index
        write_idx.store(0, std::memory_order_relaxed);
        // Free all segments
        for (std::size_t i = 0; i < max_segments; ++i) {
            Segment* seg = segments[i].exchange(nullptr, std::memory_order_acq_rel);
            if (seg) {
                delete seg;
            }
        }
    }

};

} // namespace cpu
} // namespace artea

#endif