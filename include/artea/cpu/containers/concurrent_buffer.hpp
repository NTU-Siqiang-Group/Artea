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

/*
 * @FilePath: /Artea/include/artea/cpu/containers/concurrent_buffer.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: A concurrent buffer that allows multiple threads to append elements
 *              without locks, using cache line interleaving to minimize false sharing.
 * @Alert: This implementation is buggy!!! Do not use it for now.
 */

#pragma once

#include <cstddef>

#include <artea/definitions.hpp>
#include <artea/cpu/containers/allocator.hpp>

namespace artea {
namespace cpu {

template <
    typename T,
    std::size_t buf_capacity
>
struct alignas(CACHE_LINE_SIZE) ConcurrentBuffer {

    using container_t = avx512_container_t<T>;

    /** @brief Number of elements that fit in a single cache line */
    static constexpr std::size_t num_elements_per_cacheline = CACHE_LINE_SIZE / sizeof(T);

    /**
     * @brief Number of shards (cache lines) available.
     * Calculated by dividing total capacity by elements per line.
     */
    static constexpr std::size_t num_cachelines = buf_capacity / num_elements_per_cacheline;

    /**
     * @brief Stride size.
     * This represents the jump size to get to the start of the next cache line block.
     * In this layout, it equals the capacity of a single cache line.
     */
    static constexpr std::size_t cacheline_stride = num_elements_per_cacheline;

    /**
      * --- Compile-Time Checks ---
      */

    /** @brief Force T to be smaller than or equal to a cache line.
      * This prevents division by zero in num_elements_per_cacheline calculation
      * and ensures the interleaving strategy makes sense.
      */
    static_assert(sizeof(T) <= CACHE_LINE_SIZE,
        "Error: Type T is larger than CACHE_LINE_SIZE. ConcurrentBuffer is optimized for small objects.");

    /** @brief Force buf_capacity to be a perfect multiple of elements per cache line.
      * This is CRITICAL. If not aligned, the interleaving math (row/col calculation)
      * will cause physical index collisions (different logical indices mapping to the same physical slot).
      */
    static_assert(buf_capacity % num_elements_per_cacheline == 0,
        "Error: buf_capacity must be a multiple of num_elements_per_cacheline to prevent index collision.");

    /** @brief Atomic index for writing (Monotonically increasing) */
    alignas(CACHE_LINE_SIZE) std::atomic<std::size_t> write_idx {0};

    /** @brief Bottom-level container */
    container_t container;

    /** @brief Default constructor that reserves buffer size */
    ConcurrentBuffer() {
        // Resize container to hold the physical data.
        // Note: vector memory must be physically allocated for direct access.
        container.resize(buf_capacity);
        // Ensure atomic counter starts at 0
        write_idx.store(0, std::memory_order_relaxed);
    }

    ConcurrentBuffer(ConcurrentBuffer&& other) noexcept
        : container(std::move(other.container)) {
        size_t current_idx = other.write_idx.load(std::memory_order_relaxed);
        write_idx.store(current_idx, std::memory_order_relaxed);
        other.write_idx.store(0, std::memory_order_relaxed);
    }

    /** @brief Default destructor */
    ~ConcurrentBuffer() = default;

    ConcurrentBuffer(const ConcurrentBuffer&) = delete;
    ConcurrentBuffer& operator=(const ConcurrentBuffer&) = delete;

    /**
     * @brief Tries to add an element to the buffer concurrently.
     *
     * Uses fetch_add to reserve a logical slot, then maps that logical slot
     * to a physical address interleaved across cache lines to prevent false sharing.
     *
     * @param element The element to insert.
     * @return true if insertion was successful.
     * @return false if the buffer is full.
     */
    auto append(const T& element) -> void {
        const std::size_t logical_idx = write_idx.fetch_add(1, std::memory_order_relaxed);
        const std::size_t row = logical_idx % num_cachelines;
        const std::size_t col = logical_idx / num_cachelines;

        // Calculate final flat index in the container: (CacheLine Index * Size of CacheLine) + Offset within Line
        const std::size_t physical_idx = (row * cacheline_stride) + col;
        container[physical_idx] = element;
    }

    /**
     * @brief Gets the number of elements currently written (approximate).
     * @return Number of elements. Max capped at buf_capacity.
     */
    __attribute__((always_inline))
    auto size() const -> std::size_t {
        std::size_t idx = write_idx.load(std::memory_order_relaxed);
        return idx > buf_capacity ? buf_capacity : idx;
    }

    /**
     * @brief Flushes all elements from the buffer into a vector.
     *
     * This function constructs a new vector containing all valid elements
     * in their logical insertion order. It handles the mapping from
     * logical indices back to the interleaved physical indices.
     * After flushing, the buffer's write index is reset to 0.
     *
     * @note This function assumes it is called in a thread-safe context
     * (e.g., single-consumer or externally synchronized), meaning no other
     * threads are actively calling try_add() during this operation.
     *
     * @return A std::vector<T> containing all elements.
     */
    auto flush() -> std::vector<T> {
        // Load current count with Acquire semantics to ensure we see all writes
        const std::size_t current_idx = write_idx.load(std::memory_order_acquire);

        // Cap the count at capacity (in case of overflow)
        const std::size_t count = (current_idx > buf_capacity) ? buf_capacity : current_idx;

        // Prepare result vector
        std::vector<T> result;
        result.reserve(count);

        // Iterate logically and map back to physical address
        // We must reconstruct the order: logical 0, logical 1, logical 2...
        for (std::size_t i = 0; i < count; ++i) {
            // Re-calculate the interleaving logic used in try_add
            const std::size_t row = i % num_cachelines;
            const std::size_t col = i / num_cachelines;

            // Calculate physical index
            const std::size_t physical_idx = (row * cacheline_stride) + col;

            // Move the element to the result vector
            // using std::move is efficient and supports move-only types
            result.push_back(std::move(container[physical_idx]));
        }

        // Reset the write index to 0
        // Release semantics ensure that the reset is visible to other threads
        // after the reads above are complete.
        write_idx.store(0, std::memory_order_release);

        return result;
    }

    /**
     * @brief Clears the buffer by resetting the write index.
     *
     * This effectively discards all current elements in the buffer.
     * Note that the underlying container still holds the data,
     * but it will be overwritten on subsequent insertions.
     */
    __attribute__((always_inline))
    auto clear() -> void {
        write_idx.store(0, std::memory_order_relaxed);
    }

};  // struct ConcurrentBuffer

}   // namespace cpu
}   // namespace artea