/*
 * @FilePath: /Artea/include/artea/cpu/index/flat_graph.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Date: 2026-01-31
 * @Description: Flat graph structure for graph-based index.
 */

#pragma once

#include <cstddef>
#include <vector>

namespace artea {
namespace cpu {

/**
 * @brief Flat graph structure for graph-based index.
 * @tparam IndexTraitsT The index traits type.
 */
template <typename IndexTraitsT>
class FlatGraph {

    using vertex_num_t = typename IndexTraitsT::vertex_num_t;
    using vertex_id_t = typename IndexTraitsT::vertex_id_t;
    using distance_t = typename IndexTraitsT::distance_t;
    using nbr_t = typename IndexTraitsT::nbr_t;
    using nbr_arr_t = typename IndexTraitsT::nbr_arr_t;
    using vector_array_t = typename IndexTraitsT::vector_array_t;

public:
    /**
     * @brief Construct a new Flat Graph object.
     * @param vecs_data Reference to the vector data for this layer.
     * @param num_vertices The total number of vertices in the graph.
     * @param max_nbr_size Maximum number of neighbors (for overflow control).
     * @param reserved_nbr_size The maximum number of neighbors per vertex (default: 2 * max_nbr_size = 32).
     */
    FlatGraph(
        const vector_array_t& vecs_data,
        const vertex_num_t num_vertices,
        const vertex_num_t max_nbr_size = 16,
        const vertex_num_t reserved_nbr_size = 32
    ) :
        _num_vertices(num_vertices),
        _reserved_nbr_size(reserved_nbr_size),
        _max_nbr_size(max_nbr_size),
        _vecs_data(vecs_data)
    {
        _nbrs_arr.resize(num_vertices);
        for (vertex_num_t i = 0; i < num_vertices; ++i) {
            _nbrs_arr[i].reserve(reserved_nbr_size);
        }
    }

    // Copying is deleted
    FlatGraph(const FlatGraph&) = delete;
    FlatGraph& operator=(const FlatGraph&) = delete;

    // default move constructor and assignment
    FlatGraph(FlatGraph&&) noexcept = default;
    FlatGraph& operator=(FlatGraph&&) noexcept = default;

    // --- Public Interface ---

    __attribute__((always_inline))
    auto get_num_vertices() const -> vertex_num_t {
        return _num_vertices;
    }

    __attribute__((always_inline))
    auto get_reserved_nbr_size() const -> vertex_num_t {
        return _reserved_nbr_size;
    }

    __attribute__((always_inline))
    auto get_max_nbr_size() const -> vertex_num_t {
        return _max_nbr_size;
    }

    __attribute__((always_inline))
    auto set_max_nbr_size(const vertex_num_t max_nbr_size) -> void {
        _max_nbr_size = max_nbr_size;
    }

    __attribute__((always_inline))
    auto get_nbrs_arr() -> std::vector<nbr_arr_t>& {
        return _nbrs_arr;
    }

    __attribute__((always_inline))
    auto get_nbrs_arr() const -> const std::vector<nbr_arr_t>& {
        return _nbrs_arr;
    }

    __attribute__((always_inline))
    auto fetch_nbrs(const vertex_id_t src) const -> const nbr_arr_t& {
        return _nbrs_arr[src];
    }

    __attribute__((always_inline))
    auto fetch_nbrs(const vertex_id_t src) -> nbr_arr_t& {
        return _nbrs_arr[src];
    }

    __attribute__((always_inline))
    auto get_vecs_data() const -> const vector_array_t& {
        return _vecs_data;
    }

    // --- Inter-layer edge accessors ---

    /**
     * @brief Get inter-layer edges.
     * @return Reference to inter-layer edges container.
     */
    __attribute__((always_inline))
    auto get_inter_layer_edges() -> std::vector<vertex_id_t>& {
        return _inter_layer_edges;
    }

    __attribute__((always_inline))
    auto get_inter_layer_edges() const -> const std::vector<vertex_id_t>& {
        return _inter_layer_edges;
    }

    /**
     * @brief Initialize inter-layer edges with the given size.
     * @param size The number of inter-layer edges to allocate.
     */
    auto init_inter_layer_edges(const vertex_num_t size) -> void {
        _inter_layer_edges.resize(size);
    }

protected:
    /** @brief Number of vertices in the graph. */
    vertex_num_t _num_vertices;

    /** @brief Number of (expected) neighbors per vertex. */
    vertex_num_t _reserved_nbr_size;

    /** @brief Maximum number of neighbors (for overflow control). */
    vertex_num_t _max_nbr_size;

    /** @brief Array of neighbors for each vertex. */
    std::vector<nbr_arr_t> _nbrs_arr;

    /** @brief Const reference to vector data for this layer. */
    const vector_array_t& _vecs_data;

    /** @brief Inter-layer edges: maps vertex in current layer to vertex in next layer. */
    std::vector<vertex_id_t> _inter_layer_edges;

};  // class FlatGraph

}   // namespace cpu
}   // namespace artea