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
 * @FilePath: /Artea/include/artea/cpu/propagation/op_log_table.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description:
 */

#pragma once

#include <cstddef>
#include <vector>
#include <utility>

#include <artea/cpu/containers/allocator.hpp>
#include <artea/cpu/propagation/graph_op_log.hpp>
#include <artea/definitions.hpp>

namespace artea {
namespace cpu {

template <
    typename vertex_num_t,
    typename vec_ele_t,
    typename log_buffer_t
>
class OpLogTable {

    /** @brief vertex identifier type */
    using vertex_id_t = vertex_num_t;

    /** @brief distance type */
    using distance_t = vec_ele_t;

    /** @brief graph operation type */
    using log_t = GraphOperationLog<vertex_num_t, vec_ele_t>;

    /** @brief operation container type */
    using log_container_t = typename log_buffer_t::container_t;

    /** --- Compile Time Check --- **/
    static_assert(sizeof(log_t) == 16, "GraphOperationLog size must be 16 bytes for alignment.");

public:

    /** @brief Construct a new Delegate Table object. */
    OpLogTable(const vertex_num_t num_vertices) {
        _log_bufs.resize(num_vertices);
    }

    /** @brief Try to add a delegation operation to the log table.
      * @param executor_vid The vertex executing the operation.
      * @param old_neighbor_id The old neighbor vertex to be replaced/sacrificed.
      * @param new_neighbor_id The new neighbor vertex to be added/delegated.
      * @param new_edge_dist The distance of the new edge to be added.
      * @return true if the operation was successfully added; false if the buffer is full.
     */
    __attribute__((always_inline))
    auto append_log(
        const vertex_id_t executor_vid,
        const graph_op_t op_type,
        const op_direction_t op_direction,
        const vertex_id_t old_neighbor_id,
        const vertex_id_t new_neighbor_id,
        const distance_t new_edge_dist
    ) -> void {
        #ifndef NDEBUG
        if (is_nan_distance(new_edge_dist)) {
            throw std::runtime_error("Error: Logging an operation with NaN distance is not allowed.");
        }
        #endif

        // Append the operation to the buffer
        _log_bufs[executor_vid].append(
            op_type,
            op_direction,
            old_neighbor_id,
            new_neighbor_id,
            new_edge_dist
        );
    }

    /** @brief Flush the delegation operations for a vertex.
      * @param vid The vertex identifier.
      * @return A pair of vectors containing append operations and replace operations.
     */
    /** @warning flush_logs is no longer supported */
    // __attribute__((always_inline))
    // auto flush_logs(vertex_id_t vid) -> log_container_t {
    //     log_container_t ops = _log_bufs[vid].flush();
    //     return { std::move(ops) };
    // }

    /** @brief Clear the logged operations for a vertex.
      * @param vid The vertex identifier.
      */
    __attribute__((always_inline))
    auto clear_logs(vertex_id_t vid) -> void {
        _log_bufs[vid].clear();
    }

    /** @brief Get the operation container for a vertex.
      * @param vid The vertex identifier.
      * @return Reference to the operation container.
      */
    __attribute__((always_inline))
    auto get_log_container(vertex_id_t vid) -> log_container_t& {
        return _log_bufs[vid].get_container();
    }

private:

    /** @brief Operation log buffers for each vertex. */
    cache_aligned_container_t<log_buffer_t> _log_bufs;

};  // class OpLogTable

}   // namespace cpu
}   // namespace artea