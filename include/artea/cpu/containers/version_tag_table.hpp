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
 * @FilePath: /Artea/include/artea/cpu/containers/version_tag_table.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description:
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <artea/cpu/containers/allocator.hpp>
#include <artea/cpu/router/visited_table_concept.hpp>

namespace artea {
namespace cpu {

/**
 * @brief A version-tag based visited table for thread-local use.
 *
 * Design Philosophy:
 * Instead of clearing the entire table between queries (O(n) memset),
 * this table uses a monotonically increasing version counter. Each slot
 * stores the version at which it was last marked. To "clear" the table,
 * we simply bump the version — O(1).
 *
 * When the version counter wraps around (uint16_t max = 65535), we
 * memset the entire array and reset the counter. Because each thread
 * owns its table permanently in the parallel_for, this wrap-around
 * only affects the single thread that hit the limit — no global flush.
 *
 * Satisfies the VisitedTable concept: set(idx), test(idx), clear().
 */
class VersionTagTable {

public:
    using tag_t = uint16_t;

    VersionTagTable() = default;

    explicit VersionTagTable(const size_t num_elements) {
        resize(num_elements);
    }

    ~VersionTagTable() {
        _mm_free(_tags);
    }

    // Non-copyable: single-owner semantics.
    VersionTagTable(const VersionTagTable&) = delete;
    VersionTagTable& operator=(const VersionTagTable&) = delete;

    // Movable.
    VersionTagTable(VersionTagTable&& other) noexcept
        : _tags(other._tags),
          _num_elements(other._num_elements),
          _current_version(other._current_version)
    {
        other._tags = nullptr;
        other._num_elements = 0;
        other._current_version = 0;
    }

    VersionTagTable& operator=(VersionTagTable&& other) noexcept {
        if (this != &other) {
            _mm_free(_tags);
            _tags = other._tags;
            _num_elements = other._num_elements;
            _current_version = other._current_version;
            other._tags = nullptr;
            other._num_elements = 0;
            other._current_version = 0;
        }
        return *this;
    }

    /**
     * @brief Resize and zero-initialize the tag array.
     * @param num_elements Number of elements to track.
     */
    void resize(const size_t num_elements) {
        _mm_free(_tags);
        _num_elements = num_elements;
        const size_t bytes = _num_elements * sizeof(tag_t);
        _tags = static_cast<tag_t*>(_mm_malloc(bytes, CACHE_LINE_SIZE));
        std::memset(_tags, 0, bytes);
        _current_version = 1;
    }

    /**
     * @brief Mark an element as visited in the current version.
     * @param idx The index of the element to mark.
     */
    __attribute__((always_inline))
    void set(const size_t idx) {
        _tags[idx] = _current_version;
    }

    /**
     * @brief Check if an element is visited in the current version.
     * @param idx The index of the element to check.
     * @return true if the element was visited in this version.
     */
    __attribute__((always_inline))
    bool test(const size_t idx) const {
        return _tags[idx] == _current_version;
    }

    /**
     * @brief Clear all visited marks by bumping the version counter.
     *
     * O(1) in the common case. When the version wraps around to 0,
     * we memset the array and reset — this happens once every 65535 clears,
     * and only affects this single thread-local table.
     */
    __attribute__((always_inline))
    void clear() {
        _current_version++;
        if (_current_version == 0) {
            std::memset(_tags, 0, _num_elements * sizeof(tag_t));
            _current_version = 1;
        }
    }

    /** @brief Get the number of elements tracked. */
    __attribute__((always_inline))
    size_t num_elements() const { return _num_elements; }

    /** @brief Get the current version tag. */
    __attribute__((always_inline))
    tag_t current_version() const { return _current_version; }

private:

    tag_t* _tags = nullptr;
    size_t _num_elements = 0;
    tag_t _current_version = 0;

};  // class VersionTagTable

static_assert(VisitedTable<VersionTagTable>, "VersionTagTable must satisfy VisitedTable concept");

}   // namespace cpu
}   // namespace artea
