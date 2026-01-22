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
 * @FilePath: /Artea/include/artea/common/element_pos.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2025-11-27 11:43:57
 * @Date: 2025-11-27 11:43:49
 * @Description: Common utility functions for finding element positions in small arrays.
 */

namespace artea {
namespace cpu {

/** @brief Find the position of the minimum element in a small fixed-size array.
  * @tparam T The type of the elements in the array.
  * @tparam N The size of the array (must be 2 or 3).
  * @param arr Pointer to the first element of the array.
  * @return The index of the minimum element in the array.
  * @note This function is optimized for small arrays of size 2 or 3 using compile-time branching.
 */
template <typename T, int N>
auto min_element_pos(T* arr) -> int {
    if constexpr (N == 2) {
        return (arr[0] < arr[1]) ? 0 : 1;
    }
    if constexpr (N == 3) {
        return (arr[0] < arr[1])
               ? ((arr[0] < arr[2]) ? 0 : 2)
               : ((arr[1] < arr[2]) ? 1 : 2);
    }
}

/** @brief Find the position of the maximum element in a small fixed-size array.
  * @tparam T The type of the elements in the array.
  * @tparam N The size of the array (must be 2 or 3).
  * @param arr Pointer to the first element of the array.
  * @return The index of the maximum element in the array.
  * @note This function is optimized for small arrays of size 2 or 3 using compile-time branching.
 */
template <typename T, int N>
auto max_element_pos(T* arr) -> int {
    if constexpr (N == 2) {
        return (arr[0] > arr[1]) ? 0 : 1;
    }
    if constexpr (N == 3) {
        return (arr[0] > arr[1])
                ? ((arr[0] > arr[2]) ? 0 : 2)
                : ((arr[1] > arr[2]) ? 1 : 2);
    }
}

} // namespace cpu
} // namespace artea