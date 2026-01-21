/*
 * @FilePath: /Artea/include/artea/cpu/index/index_graph.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2026-01-18 14:45:17
 * @Date: 2025-10-17 15:32:18
 * @Description:
 */

#pragma once

#include <cstddef>
#include <stdexcept>
#include <variant>
#include <cstdint>
#include <vector>

#include <tbb/parallel_for.h>

#include <artea/common/definitions.hpp>
#include <artea/cpu/utils/direction.hpp>
#include <artea/cpu/index/neighbor.hpp>
#include <artea/cpu/containers/allocator.hpp>
#include <artea/cpu/propagation/graph_op_log.hpp>

namespace artea {
namespace cpu {

/**
 * @brief The index graph class.
 * @tparam{vertex_num_t} The type of vertex number.
 * @tparam{vec_ele_t} The type of vector element.
 * @tparam{direction} The direction of the graph (IN, OUT, HIBRID).
 */
template <typename BaseTraitsT>
class IndexGraph {

    using vertex_num_t = typename BaseTraitsT::vertex_num_t;
    using vertex_id_t = typename BaseTraitsT::vertex_id_t;
    using distance_t = typename BaseTraitsT::distance_t;
    using nbr_t = typename BaseTraitsT::nbr_t;
    using nbr_arr_t = typename BaseTraitsT::nbr_arr_t;

public:
    /**
     * @brief Construct a new Index Graph object with aligned memory.
     * @param num_vertices The total number of vertices in the graph.
     * @param reserved_nbrs_size The maximum number of (in-/out-) neighbors per vertex.
     */
    IndexGraph(
        const vertex_num_t num_vertices,
        const vertex_num_t reserved_nbrs_size
    ) :
        _num_vertices(num_vertices),
        _reserved_nbrs_size(reserved_nbrs_size)
    {
        _nbrs_arr.resize(num_vertices);
        for (vertex_num_t i = 0; i < num_vertices; ++i) {
            _nbrs_arr[i].reserve(reserved_nbrs_size);
        }
    }

    IndexGraph() = default;

    ~IndexGraph() = default;

    // default move constructor and assignment
    IndexGraph(IndexGraph&&) noexcept = default;
    IndexGraph& operator=(IndexGraph&&) noexcept = default;

    // Copying is deleted
    IndexGraph(const IndexGraph&) = delete;
    IndexGraph& operator=(const IndexGraph&) = delete;

    // --- Accessors ---

    __attribute__((always_inline))
    auto get_num_vertices() const -> vertex_num_t {
        return _num_vertices;
    }

    __attribute__((always_inline))
    auto get_reserved_nbrs_size() const -> vertex_num_t {
        return _reserved_nbrs_size;
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

protected:

    /** @brief Number of vertices in the graph. */
    vertex_num_t _num_vertices;

    /** @brief Number of (expected) neighbors per vertex. */
    vertex_num_t _reserved_nbrs_size;

    /** @brief Array of neighbors for each vertex. */
    std::vector<nbr_arr_t> _nbrs_arr;

};  // class IndexGraph

}   // namespace cpu
}   // namespace artea
