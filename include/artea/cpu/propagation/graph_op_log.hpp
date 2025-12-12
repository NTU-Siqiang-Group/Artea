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
 * @FilePath: /Artea/include/artea/cpu/propagation/graph_op_log.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description:
 */

#pragma once

#include <artea/definitions.hpp>
#include <artea/cpu/utils/direction.hpp>

namespace artea {
namespace cpu {

enum class graph_op_t : uint8_t {
    APPEND = 0,
    REPLACE = 1
};

template <typename vertex_num_t, typename vec_ele_t>
struct alignas(16) GraphOperationLog {

    using vertex_id_t = vertex_num_t;
    using distance_t = vec_ele_t;

    /** @brief The type of graph operation (APPEND/REPLACE). */
    graph_op_t op_type : 8;           // 2 bytes

    /** @brief The direction of the operation (IN/OUT). */
    op_direction_t op_direction: 8;   // 2 bytes

    /** @brief The old neighbor vertex to be replaced/sacrificed.
      * @note old_neighbor_id == invalid_vertex_id indicates an append operation.
      */
    vertex_id_t old_neighbor_id: 32;  // 4 bytes

    /** @brief The new neighbor vertex to be added/delegated.
      * @note new_neighbor_id == invalid_vertex_id indicates a removal operation.
      */
    vertex_id_t new_neighbor_id: 32;  // 4 bytes

    /** @brief The distance from executor vertex to the new neighbor */
    distance_t new_edge_dist;         // 4 bytes

    GraphOperationLog() = default;

    GraphOperationLog(
        const graph_op_t op_type,
        const op_direction_t op_direction,
        const vertex_id_t old_neighbor_id,
        const vertex_id_t new_neighbor_id,
        const distance_t new_edge_dist
    ) :
        op_type(op_type),
        op_direction(op_direction),
        old_neighbor_id(old_neighbor_id),
        new_neighbor_id(new_neighbor_id),
        new_edge_dist(new_edge_dist)
    {}

};  // struct GraphOperationLog

}   // namespace cpu
}   // namespace artea