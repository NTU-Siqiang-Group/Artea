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

#pragma once

#include <vector>
#include <limits>
#include <algorithm>
#include <functional>

namespace artea {
namespace cpu {

/**
 * @brief 4-ary Max-Heap with Sentinel Padding (std::priority_queue-compatible semantics).
 *
 * Follows std::priority_queue conventions:
 * - Max-heap by default (largest element at top, via Compare = std::less<T>).
 * - top() returns a const reference to the top element.
 * - pop() removes the top element without returning it.
 *
 * Optimization Strategy:
 * Appends 3 sentinel values at the end of the underlying container.
 * This guarantees that access to children [4*i+1 ... 4*i+4] is ALWAYS valid memory.
 *
 * Mechanism:
 * 1. Physical Size >= Logical Size + 3.
 * 2. Slots [_logical_size ... _logical_size + 3] always contain '_sentinel_entry'.
 * 3. Sentinel must NEVER win a comparison against any valid element.
 *    For max-heap (Compare = std::less<T>): sentinel should be the minimum possible value.
 *
 * Result:
 * Completely removes boundary checks (Slow Path) for individual children in sift_down.
 *
 * @tparam T The element type.
 * @tparam ContainerT The underlying container type (must support random access and resize).
 * @tparam Compare Comparator type. Default std::less<T> gives max-heap (same as std::priority_queue).
 */
template <typename T, typename ContainerT, typename Compare = std::less<T>>
class FourAryHeap {
public:
    using entry_t = T;
    using container_t = ContainerT;
    static constexpr Compare comparator = Compare();

    static constexpr uint32_t Arity = 4;
    static constexpr size_t Padding = 3; // 4-ary heap needs 3 sentinels

    /**
     * @brief Constructor with sentinel value.
     * @param sentinel_value The sentinel value. For max-heap (std::less), this must be the
     *        minimum possible value so that sentinels never win comparisons against valid elements.
     */
    explicit FourAryHeap(const entry_t& sentinel_value)
        : _sentinel_entry(sentinel_value) {
        _data.reserve(64);
        _data.resize(Padding, _sentinel_entry);
    }

    __attribute__((always_inline))
    auto reserve(size_t capacity) -> void {
        _data.reserve(capacity + Padding);
    }

    __attribute__((always_inline))
    auto empty() const -> bool {
        return _logical_size == 0;
    }

    __attribute__((always_inline))
    auto size() const -> size_t {
        return _logical_size;
    }

    __attribute__((always_inline))
    auto clear() -> void {
        _logical_size = 0;
        ensure_padding();
    }

    /** @brief Access the top element (the largest under Compare = std::less). */
    __attribute__((always_inline))
    auto top() const -> const entry_t& {
        return _data[0];
    }

    /**
     * @brief Push a new value into the heap.
     */
    __attribute__((always_inline))
    auto push(const entry_t& val) -> void {
        if (_logical_size + 1 + Padding > _data.size()) {
            size_t new_cap = std::max(_data.size() * 2, size_t(64));
            new_cap = std::max(new_cap, _logical_size + 1 + Padding);
            _data.resize(new_cap, _sentinel_entry);
        }

        _data[_logical_size] = val;
        sift_up(_logical_size);
        _logical_size++;
    }

    /**
     * @brief Remove the top element (does not return it; use top() first).
     */
    __attribute__((always_inline))
    auto pop() -> void {
        _logical_size--;
        _data[0] = _data[_logical_size];

        // [CRITICAL] Restore sentinel at the vacated slot
        _data[_logical_size] = _sentinel_entry;

        if (_logical_size > 0) {
            sift_down(0);
        }
    }

    /**
     * @brief Batch initialize from a vector using Floyd's heap construction.
     * @param input The input vector of elements.
     * @complexity O(N)
     */
    auto initialize(const std::vector<entry_t>& input) -> void {
        _logical_size = input.size();
        _data.assign(input.begin(), input.end());
        _data.resize(_logical_size + Padding, _sentinel_entry);

        // Floyd's heap construction (sift-down from the last parent)
        if (_logical_size > 1) {
            for (int i = (static_cast<int>(_logical_size) - 2) / Arity; i >= 0; --i) {
                sift_down(static_cast<size_t>(i));
            }
        }
    }

private:
    container_t _data;
    size_t _logical_size = 0;
    entry_t _sentinel_entry;

    __attribute__((always_inline))
    auto ensure_padding() -> void {
        if (_data.size() < _logical_size + Padding) {
            _data.resize(_logical_size + Padding, _sentinel_entry);
        } else {
            for (size_t i = 0; i < Padding; ++i) {
                _data[_logical_size + i] = _sentinel_entry;
            }
        }
    }

    /**
     * @brief Sift up: move a node towards the root if it wins the comparison against its parent.
     *
     * For max-heap (Compare = std::less<T>):
     *   comparator(parent, child) == true means parent < child, so child should go up.
     */
    __attribute__((always_inline))
    auto sift_up(size_t index) -> void {
        entry_t target = _data[index];
        while (index > 0) {
            size_t parent_idx = (index - 1) / Arity;
            // If parent loses to target under Compare, swap upward
            if (comparator(_data[parent_idx], target)) {
                _data[index] = _data[parent_idx];
                index = parent_idx;
            } else {
                break;
            }
        }
        _data[index] = target;
    }

    /**
     * @brief Optimized sift_down with MINIMAL BOUNDARY CHECKS.
     *
     * Because _data is padded with sentinel values, accessing child_start + 1/2/3
     * is ALWAYS valid memory. Sentinels never win comparisons against valid elements,
     * so the algorithm naturally ignores them without explicit boundary checks.
     *
     * For max-heap (Compare = std::less<T>):
     *   We find the child that beats all others, then check if it beats the target.
     */
    __attribute__((always_inline))
    auto sift_down(size_t index) -> void {
        entry_t target = _data[index];

        while (true) {
            size_t child_start = Arity * index + 1;
            if (__builtin_expect(child_start >= _logical_size, 0)) { break; }

            size_t idx0 = child_start;
            size_t idx1 = child_start + 1;
            size_t idx2 = child_start + 2;
            size_t idx3 = child_start + 3;

            // Tree-reduction: find the "winning" child (the one that should be on top)
            // comparator(a, b) == true means b wins over a
            size_t best_01 = comparator(_data[idx0], _data[idx1]) ? idx1 : idx0;
            size_t best_23 = comparator(_data[idx2], _data[idx3]) ? idx3 : idx2;
            size_t best_child_idx = comparator(_data[best_01], _data[best_23]) ? best_23 : best_01;

            // If the best child wins over target, swap down
            if (comparator(target, _data[best_child_idx])) {
                _data[index] = _data[best_child_idx];
                index = best_child_idx;
            } else {
                break;
            }
        }
        _data[index] = target;
    }
};

} // namespace cpu
} // namespace artea
