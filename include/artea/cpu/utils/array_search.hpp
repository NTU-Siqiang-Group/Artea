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
 * @FilePath: /Artea/include/artea/cpu/utils/array_search.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description:
 */

#pragma once

#include <cstddef>
#include <functional>
#include <vector>



namespace artea {
namespace cpu {

enum class SearchMethodT : uint8_t {
    BINARY_SEARCH = 0,
    LINEAR_SEARCH = 1
};

/**
 * @brief Perform search to find the index for a target value in a sorted array.
 * @tparam T The type of elements in the array.
 * @tparam array_index_t The type used for indexing.
 * @param arr The sorted array.
 * @param target The target value to search for.
 * @param compare_func A comparison function that returns true if the first argument is less than the second.
 * @return The index of the target if found, otherwise invalid_vertex_id().
 */
template <
    typename T,
    typename container_t,
    typename array_index_t,
    SearchMethodT method,
    typename compare_func_t
>
auto array_search(
    const container_t& arr,
    const T target,
    const compare_func_t compare_func
) -> array_index_t {

    array_index_t arr_size = static_cast<array_index_t>(arr.size());

    // BINARY_SEARCH (Branchless Implementation, can not be used for unsorted arrays)
    if constexpr (method == SearchMethodT::BINARY_SEARCH) {
        array_index_t base = 0;
        array_index_t len = arr_size;

        while (len > 0) {
            array_index_t half = len >> 1;
            array_index_t mid = base + half;

            // Standard Binary Search Logic:
            // if (arr[mid] < target) { base = mid + 1; len = len - half - 1; }
            // else { len = half; }

            // Branchless Optimization:
            // We use the result of the comparison to calculate the new base and len.
            // Compilers will optimize this ternary operator into CMOV (Conditional Move) instructions,
            // avoiding CPU pipeline flushes caused by branch misprediction.
            bool go_right = compare_func(arr[mid], target);

            base = go_right ? mid + 1 : base;
            len  = go_right ? len - (half + 1) : half;
        }

        // 'base' is now the lower_bound (first element >= target).
        // Check if index is valid AND if arr[base] == target.
        // Since arr[base] >= target is guaranteed by lower_bound logic,
        // we only need to check if target < arr[base] is FALSE.
        if (base < arr_size && !compare_func(target, arr[base])) {
            return base;
        }
        return invalid_vertex_id<array_index_t>();
    }
    // LINEAR_SEARCH (can be used for unsorted arrays as well)
    else if constexpr (method == SearchMethodT::LINEAR_SEARCH) {
        for (array_index_t i = 0; i < arr_size; ++i) {
            if (!compare_func(arr[i], target) && !compare_func(target, arr[i])) {
                return i;
            }
        }
        return invalid_vertex_id<array_index_t>();
    }
}

}   // namespace cpu
}   // namespace artea