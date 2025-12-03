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
 * @FilePath: /Artea/include/artea/cpu/index/neighbor.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Neighbor structure definition.
 */

#pragma once

namespace artea {
namespace cpu {

/** @brief Neighbor structure for storing vertex ID and distance. */
template <typename vertex_num_t, typename vec_ele_t>
struct alignas(8) Neighbor {   // 8 bytes

    using distance_t = vec_ele_t;
    using vertex_id_t = vertex_num_t;

    vertex_id_t dest;
    distance_t distance;
    // bool new_flag;
    // uint8_t padding[7];

};  // struct Neighbor

/** @brief Comparator for Neighbor, lambda function. */
template <typename vertex_num_t, typename vec_ele_t>
constexpr auto NeighborComparator = [](
    const Neighbor<vertex_num_t, vec_ele_t>& a,
    const Neighbor<vertex_num_t, vec_ele_t>& b
) -> bool {
    return a.distance < b.distance or
          (a.distance == b.distance && a.dest < b.dest);
};  // constexpr auto NeighborComparator

}   // namespace cpu
}   // namespace artea