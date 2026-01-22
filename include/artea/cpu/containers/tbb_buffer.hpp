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
 * @FilePath: /Artea/include/artea/cpu/containers/tbb_buffer.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description:
 */

#pragma once

#include <vector>
#include <algorithm>
#include <tbb/concurrent_vector.h>
#include <artea/common/logger.hpp>

namespace artea {
namespace cpu {

/**
 * @brief A single-container concurrent buffer based on TBB's concurrent_vector.
 *
 * @tparam T The type of elements stored.
 * @tparam BufCapacity Initial reservation size (optimization hint).
 */
template <
    typename T,
    std::size_t BufCapacity = 32
>
struct alignas(CACHE_LINE_SIZE) TbbBuffer {

    using container_t = tbb::concurrent_vector<T>;

    // -------------------------------------------------------------------------
    // Member Variables
    // -------------------------------------------------------------------------

    /** @brief The underlying TBB container */
    container_t container;

    // -------------------------------------------------------------------------
    // Constructor / Destructor
    // -------------------------------------------------------------------------

    /**
     * @brief Construct a new Tbb Buffer object.
     * Reserves memory to avoid early allocation overhead.
     */
    TbbBuffer() {
        // TBB vector supports reserve to pre-allocate memory segments.
        // This makes the performance closer to a fixed-size buffer for the first N elements.
        container.reserve(BufCapacity);
    }

    ~TbbBuffer() = default;

    // -------------------------------------------------------------------------
    // API Methods (Matching ConcurrentBuffer)
    // -------------------------------------------------------------------------

    /**
     * @brief Tries to add an element to the buffer.
     *
     * @param element The element to add.
     * @return true Always returns true (TBB vector expands automatically).
     *              Returns false only if system is out of memory (std::bad_alloc).
     */
    __attribute__((always_inline))
    auto append(const T& element) -> void {
        container.push_back(element);
    }

    /**
     * @brief Emplace variant for efficiency (API enhancement).
     */
    template <typename... Args>
    __attribute__((always_inline))
    auto append(Args&&... args) -> void {
        container.emplace_back(std::forward<Args>(args)...);
    }

    __attribute__((always_inline))
    auto atomic_flush_to(std::vector<T>& dest) -> void {
        logger.error("TbbBuffer does not support atomic_flush_to operation.");
        throw std::runtime_error("atomic_flush_to is not supported in TbbBuffer.");
    }

    /**
     * @brief Get the current number of elements with locking.
     * @note This method cannot be called concurrently with append.
     */
    __attribute__((always_inline))
    auto get_container() -> container_t& {
        return container;
    }

    /**
     * @brief Flushes all elements to a standard std::vector.
     *
     * This operation is not thread-safe with respect to concurrent writers.
     * It should be called when insertion is paused or finished.
     *
     * @return std::vector<T> containing all elements in insertion order.
     */
    auto flush() -> std::vector<T> {
        // 1. Copy data: tbb::concurrent_vector iterators work with std::vector ctor
        std::vector<T> result;

        // Optimization: reserve size if known
        // Note: container.size() in TBB is not O(1), but O(segments). It's fast enough.
        result.reserve(container.size());

        // Copy elements (or move if we implemented move iterators, but standard copy is safe)
        result.assign(container.begin(), container.end());

        // 2. Clear: Reset TBB vector for reuse.
        // TBB keeps the allocated capacity segments, so next fills are fast.
        container.clear();

        return result;
    }

    /**
     * @brief Get the current number of elements.
     * @note In TBB, size() is not strictly O(1) but it is thread-safe.
     */
    __attribute__((always_inline))
    auto size() const -> std::size_t {
        return container.size();
    }

    /**
     * @brief Reset the buffer (Clear content) with locking.
     * @note This method cannot be called concurrently with append.
     */
    __attribute__((always_inline))
    auto clear() -> void {
        container.clear();
    }

};  // struct TbbBuffer

}   // namespace cpu
}   // namespace artea