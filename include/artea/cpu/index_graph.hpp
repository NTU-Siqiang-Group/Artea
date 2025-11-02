/*
 * @FilePath: /Artea/include/artea/cpu/index_graph.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2025-11-01 19:42:14
 * @Date: 2025-10-17 15:32:18
 * @Description: 
 */

#pragma once

#include <immintrin.h> // For _mm_malloc and _mm_free

namespace artea {
namespace cpu {

template <
    typename vertex_num_t,
    typename vec_ele_t,
    typename distance_t = vec_ele_t,
    typename vertex_id_t = vertex_num_t
>
class IndexGraph {

public:
    IndexGraph(vertex_num_t num_vertices, vertex_num_t num_nbrs_per_vertex) : 
        _num_vertices(num_vertices),
        _nbrs(nullptr),
        _nbrs_dists(nullptr) {}

    ~IndexGraph() {
        if (_nbrs != nullptr)
            delete[] _nbrs;
        if (_nbrs_dists != nullptr)
            delete[] _nbrs_dists;
        // if (_csr_offsets != nullptr)
        //     delete[] _csr_offsets;

        _nbrs = nullptr;
        _nbrs_dists = nullptr;
        // _csr_offsets = nullptr;
    }

    __attribute__((always_inline))
    auto get_nbrs(vertex_id_t vid) -> vertex_num_t* {
        return _nbrs + vid * _num_nbrs_per_vertex;
    }

    __attribute__((always_inline))
    auto get_num_vertices() -> vertex_num_t {
        return _num_vertices;
    }

    __attribute__((always_inline))
    auto get_num_nbrs_per_vertex() -> vertex_num_t {
        return _num_nbrs_per_vertex;
    }

private:
    /** @brief Array of neighbors. */
    vertex_num_t* _nbrs;
    
    /** @brief Array of neighbor distances. */
    distance_t* _nbrs_dists;
    
    /** @brief Number of vertices in the graph. */
    vertex_num_t _num_vertices;

    // /** @brief CSR offset array. 
    //   * This array has _num_vertices + 1 elements.
    //   * Only used by reversed graph (reversed graph may have different number of neighbors)
    // */
    // vertex_num_t* _csr_offsets;

};  // class IndexGraph

}   // namespace cpu
}   // namespace artea