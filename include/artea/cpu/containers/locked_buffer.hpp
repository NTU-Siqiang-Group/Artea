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
 * @FilePath: /Artea/include/artea/cpu/containers/locked_buffer.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description:
 */

#pragma once

#include <vector>
#include <mutex>
#include <algorithm>
#include <utility>
#include <cstddef>

#include <tbb/spin_mutex.h>

#include <artea/common/definitions.hpp>
#include <artea/cpu/containers/allocator.hpp>

namespace artea {
namespace cpu {

/**
 * @brief A single, thread-safe buffer protected by a mutex.
 *
 * @tparam T The type of elements stored.
 * @tparam container_t The underlying container type (default: std::vector<T>).
 * @tparam BufCapacity Initial reservation size (optimization hint).
 */
template <
    typename T,
    std::size_t BufCapacity,
    typename LockT = tbb::spin_mutex
>
struct alignas(CACHE_LINE_SIZE) LockedBuffer {

    using container_t = std::vector<T>;

    // -------------------------------------------------------------------------
    // Member Variables
    // -------------------------------------------------------------------------

    /** @brief The underlying container */
    container_t container;

    /** @brief The mutex protecting the container */
    // mutable allows locking in const methods like size()
    mutable LockT mtx;

    // char padding[CACHE_LINE_SIZE - sizeof(container_t) - sizeof(LockT)];

    // -------------------------------------------------------------------------
    // Constructor / Destructor
    // -------------------------------------------------------------------------

    /**
     * @brief Construct a new Locked Buffer object.
     */
    LockedBuffer() {
        // Pre-reserve memory to minimize reallocation inside the lock
        if (BufCapacity > 0) {
            container.reserve(BufCapacity);
        }
    }

    /**
     * @brief Move constructor.
     * Required for std::vector<LockedBuffer> resizing.
     * The mutex is NOT moved (it's not movable). A new mutex is created for the new object.
     * We safely move the container content.
     */
    LockedBuffer(LockedBuffer&& other) noexcept {
        // Lock the source to be safe, although in vector resize scenarios 'other'
        // is typically not accessed concurrently.
        // std::lock_guard<LockT> lock(other.mtx);
        container = std::move(other.container);
    }

    /**
     * @brief Deleted Copy Constructor and Assignment.
     * Copying a locked buffer including its lock state is semantically ambiguous and dangerous.
     */
    LockedBuffer(const LockedBuffer&) = delete;
    LockedBuffer& operator=(const LockedBuffer&) = delete;
    LockedBuffer& operator=(LockedBuffer&&) = delete;

    ~LockedBuffer() = default;

    // -------------------------------------------------------------------------
    // API Methods (Matching ConcurrentBuffer/TbbBuffer)
    // -------------------------------------------------------------------------

    /**
     * @brief Tries to add an element to the buffer with locking.
     *
     * @param element The element to add.
     * @return true Always returns true (std::vector expands automatically).
     */
    __attribute__((always_inline))
    auto append(const T& element) -> void {
        std::lock_guard<LockT> lock(mtx);
        container.push_back(element);
    }

    /**
     * @brief Emplace variant for efficiency.
     */
    template <typename... Args>
    __attribute__((always_inline))
    auto append(Args&&... args) -> void {
        std::lock_guard<LockT> lock(mtx);
        container.emplace_back(std::forward<Args>(args)...);
    }

    __attribute__((always_inline))
    auto atomic_flush_to(std::vector<T>& swap_buffer) -> void {
        std::lock_guard<LockT> lock(mtx);
        {
            std::swap(swap_buffer, container);
        }
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
     * @brief Get the current number of elements with locking.
     * @note This method cannot be called concurrently with append.
     */
    __attribute__((always_inline))
    auto size() const -> std::size_t {
        return container.size();
    }

    /**
     * @brief Flushes all elements to a standard std::vector.
     * Copies elements out and clears the internal buffer while holding the lock.
     * @note This method can be not be called concurrently with append.
     * @return std::vector<T> containing all elements in insertion order.
     */
    auto flush() -> std::vector<T> {
        std::vector<T> result;

        // Optimization: reserve destination to avoid reallocs
        result.reserve(container.size());
        // Copy data
        result.assign(container.begin(), container.end());
        // Reset internal buffer, capacity is preserved
        container.clear();

        return result;
    }

    /**
     * @brief Reset the buffer (Clear content) with locking.
     * @note This method cannot be called concurrently with append.
     */
    __attribute__((always_inline))
    auto clear() -> void {
        container.clear();
    }

};  // struct LockedBuffer

}   // namespace cpu
}   // namespace artea