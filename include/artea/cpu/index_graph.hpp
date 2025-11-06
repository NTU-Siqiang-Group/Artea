/*
 * @FilePath: /Artea/include/artea/cpu/index_graph.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2025-11-06 14:19:44
 * @Date: 2025-10-17 15:32:18
 * @Description: Refactored IndexGraph using the custom Array class for memory management.
 */

#pragma once

#include <cstddef>
#include <stdexcept>
#include <variant>

#include <artea/cpu/array.hpp> // Include our new Array class

namespace artea {
namespace cpu {

template <typename vertex_num_t, typename vec_ele_t>
struct Neighbor {   // 8 bytes

    using distance_t = vec_ele_t;
    using vertex_id_t = vertex_num_t;

    vertex_id_t dest;
    distance_t distance;
};  // struct Neighbor

template <typename vertex_num_t, typename vec_ele_t>
class IndexGraph {

public:

    using distance_t = vec_ele_t;
    using vertex_id_t = vertex_num_t;
    using nbr_t = Neighbor<vertex_num_t, vec_ele_t>;
    using nbr_arr_t = Array<nbr_t>;

    /**
     * @brief Construct a new Index Graph object with aligned memory.
     * @param num_vertices The total number of vertices in the graph.
     * @param num_nbrs_per_vertex The fixed number of neighbors for each vertex.
     */
    IndexGraph(const vertex_num_t& num_vertices, const vertex_num_t& num_nbrs_per_vertex) : 
        _num_vertices(num_vertices),
        _num_nbrs_per_vertex(num_nbrs_per_vertex) {}

    ~IndexGraph() = default;

    // The default move constructor and assignment are correct thanks to Array's move semantics.
    IndexGraph(IndexGraph&&) noexcept = default;
    IndexGraph& operator=(IndexGraph&&) noexcept = default;

    // Copying is deleted because our underlying Array is non-copyable.
    IndexGraph(const IndexGraph&) = delete;
    IndexGraph& operator=(const IndexGraph&) = delete;

    // --- Accessors ---

    __attribute__((always_inline))
    auto get_num_vertices() const -> vertex_num_t {
        return _num_vertices;
    }

    __attribute__((always_inline))
    auto get_num_nbrs_per_vertex() const -> vertex_num_t {
        return _num_nbrs_per_vertex;
    }

    // // --- Graph Operations ---
    
    // virtual auto append_nbr(const vertex_id_t& src, const nbr_t& nbr) -> void = 0;

    // virtual auto fetch_nbrs(const vertex_id_t& src) -> nbr_arr_t& = 0;

protected:
    
    /** @brief Number of vertices in the graph. */
    const vertex_num_t& _num_vertices;

    /** @brief Number of neighbors per vertex (used as stride for indexing). */
    const vertex_num_t& _num_nbrs_per_vertex;

};  // class IndexGraph

}   // namespace cpu
}   // namespace artea
