/*
 * @FilePath: /Artea/include/artea/cpu/index/compact_descent_graph.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Date: 2026-02-05
 * @Description: Compact descent graph with CSR format for efficient neighbor access.
 */

#pragma once

#include <cstddef>
#include <vector>
#include <span>

namespace artea {
namespace cpu {
namespace compact {

/**
 * @brief Compact descent graph using CSR (Compressed Sparse Row) format for efficient neighbor access.
 * @tparam IndexTraitsT The index traits type.
 */
template <typename IndexTraitsT>
class DescentGraph {

    using vertex_num_t = typename IndexTraitsT::vertex_num_t;
    using vertex_id_t = typename IndexTraitsT::vertex_id_t;
    using distance_t = typename IndexTraitsT::distance_t;
    using vector_array_t = typename IndexTraitsT::vector_array_t;
    using csr_vids_t = typename IndexTraitsT::csr_vids_t;

public:
    /**
     * @brief Construct a new Flat Search Graph object.
     * @param vecs_data Reference to the vector data for this graph.
     * @param extracted_nbr_size Fixed number of neighbors per vertex.
     */
    DescentGraph(
        const vector_array_t& vecs_data,
        const vertex_num_t extracted_nbr_size
    ) :
        _num_vertices(vecs_data.get_num_vecs()),
        _extracted_nbr_size(extracted_nbr_size),
        _vecs_data(vecs_data)
    {
        // Allocate CSR storage: num_vertices * extracted_nbr_size
        // Note: AlignedAllocator::construct skips zero-init for trivial types,
        // so resize() only allocates without the single-threaded zero-fill overhead.
        _csr_nbrs.resize(static_cast<size_t>(_num_vertices) * extracted_nbr_size);
    }

    // Copying is deleted
    DescentGraph(const DescentGraph&) = delete;
    DescentGraph& operator=(const DescentGraph&) = delete;

    // default move constructor and assignment
    DescentGraph(DescentGraph&&) noexcept = default;
    DescentGraph& operator=(DescentGraph&&) noexcept = default;

    // --- Public Interface ---

    __attribute__((always_inline))
    auto get_num_vertices() const -> vertex_num_t {
        return _num_vertices;
    }

    __attribute__((always_inline))
    auto get_extracted_nbr_size() const -> vertex_num_t {
        return _extracted_nbr_size;
    }

    /**
     * @brief Get the neighbors of a vertex.
     * @param src The source vertex ID.
     * @return Pointer to the beginning of the neighbor list.
     */
    __attribute__((always_inline))
    auto get_neighbors(const vertex_id_t src) const -> const vertex_id_t* {
        return &_csr_nbrs[static_cast<size_t>(src) * _extracted_nbr_size];
    }

    /**
     * @brief Get mutable neighbors of a vertex.
     * @param src The source vertex ID.
     * @return Pointer to the beginning of the neighbor list.
     */
    __attribute__((always_inline))
    auto get_neighbors(const vertex_id_t src) -> vertex_id_t* {
        return &_csr_nbrs[static_cast<size_t>(src) * _extracted_nbr_size];
    }

    /**
     * @brief Get the neighbors of a vertex as a std::span (const version).
     * @param src The source vertex ID.
     * @return A span over the neighbor list.
     */
    __attribute__((always_inline))
    auto fetch_nbrs(const vertex_id_t src) const -> std::span<const vertex_id_t> {
        return std::span<const vertex_id_t>(
            &_csr_nbrs[static_cast<size_t>(src) * _extracted_nbr_size],
            _extracted_nbr_size
        );
    }

    /**
     * @brief Get mutable neighbors of a vertex as a std::span (non-const version).
     * @param src The source vertex ID.
     * @return A mutable span over the neighbor list.
     */
    __attribute__((always_inline))
    auto fetch_nbrs(const vertex_id_t src) -> std::span<vertex_id_t> {
        return std::span<vertex_id_t>(
            &_csr_nbrs[static_cast<size_t>(src) * _extracted_nbr_size],
            _extracted_nbr_size
        );
    }

    __attribute__((always_inline))
    auto get_csr_nbrs() -> csr_vids_t& {
        return _csr_nbrs;
    }

    __attribute__((always_inline))
    auto get_csr_nbrs() const -> const csr_vids_t& {
        return _csr_nbrs;
    }

    __attribute__((always_inline))
    auto get_vecs_data() const -> const vector_array_t& {
        return _vecs_data;
    }

private:
    /** @brief Number of vertices in the graph. */
    vertex_num_t _num_vertices;

    /** @brief Fixed number of neighbors per vertex. */
    vertex_num_t _extracted_nbr_size;

    /** @brief CSR format neighbor storage: dense, cache-aligned array. */
    csr_vids_t _csr_nbrs;

    /** @brief Const reference to vector data for this graph. */
    const vector_array_t& _vecs_data;

};  // class DescentGraph

}   // namespace compact
}   // namespace cpu
}   // namespace artea
