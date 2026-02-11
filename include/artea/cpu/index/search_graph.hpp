/*
 * @FilePath: /Artea/include/artea/cpu/index/search_graph.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Date: 2026-02-05
 * @Description: Search graph with CSR format for efficient neighbor access.
 */

#pragma once

#include <cstddef>
#include <vector>
#include <span>
#include <stdexcept>
#include <algorithm>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

namespace artea {
namespace cpu {

/**
 * @brief Search graph using CSR (Compressed Sparse Row) format for efficient neighbor access.
 * @tparam IndexTraitsT The index traits type.
 */
template <typename IndexTraitsT>
class SearchGraph {

    using vertex_num_t = typename IndexTraitsT::vertex_num_t;
    using vertex_id_t = typename IndexTraitsT::vertex_id_t;
    using distance_t = typename IndexTraitsT::distance_t;
    using vector_array_t = typename IndexTraitsT::vector_array_t;
    using csr_graph_t = typename IndexTraitsT::csr_graph_t;
    using flat_graph_t = typename IndexTraitsT::flat_graph_t;

public:
    /**
     * @brief Construct a new Search Graph object.
     * @param num_vertices The total number of vertices in the graph.
     * @param fix_nbr_size Fixed number of neighbors per vertex.
     * @param vecs_data Reference to the vector data for this graph.
     */
    SearchGraph(
        const vertex_num_t num_vertices,
        const vertex_num_t fix_nbr_size,
        const vector_array_t& vecs_data
    ) :
        _num_vertices(num_vertices),
        _fix_nbr_size(fix_nbr_size),
        _vecs_data(vecs_data)
    {
        // Allocate CSR storage: num_vertices * fix_nbr_size
        _csr_nbrs.resize(static_cast<size_t>(num_vertices) * fix_nbr_size);
    }

    // Copying is deleted
    SearchGraph(const SearchGraph&) = delete;
    SearchGraph& operator=(const SearchGraph&) = delete;

    // default move constructor and assignment
    SearchGraph(SearchGraph&&) noexcept = default;
    SearchGraph& operator=(SearchGraph&&) noexcept = default;

    // --- Public Interface ---

    __attribute__((always_inline))
    auto get_num_vertices() const -> vertex_num_t {
        return _num_vertices;
    }

    __attribute__((always_inline))
    auto get_fix_nbr_size() const -> vertex_num_t {
        return _fix_nbr_size;
    }

    /**
     * @brief Get the neighbors of a vertex.
     * @param src The source vertex ID.
     * @return Pointer to the beginning of the neighbor list.
     */
    __attribute__((always_inline))
    auto get_neighbors(const vertex_id_t src) const -> const vertex_id_t* {
        return &_csr_nbrs[static_cast<size_t>(src) * _fix_nbr_size];
    }

    /**
     * @brief Get mutable neighbors of a vertex.
     * @param src The source vertex ID.
     * @return Pointer to the beginning of the neighbor list.
     */
    __attribute__((always_inline))
    auto get_neighbors(const vertex_id_t src) -> vertex_id_t* {
        return &_csr_nbrs[static_cast<size_t>(src) * _fix_nbr_size];
    }

    /**
     * @brief Get the neighbors of a vertex as a std::span (const version).
     * @param src The source vertex ID.
     * @return A span over the neighbor list.
     */
    __attribute__((always_inline))
    auto fetch_nbrs(const vertex_id_t src) const -> std::span<const vertex_id_t> {
        return std::span<const vertex_id_t>(
            &_csr_nbrs[static_cast<size_t>(src) * _fix_nbr_size],
            _fix_nbr_size
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
            &_csr_nbrs[static_cast<size_t>(src) * _fix_nbr_size],
            _fix_nbr_size
        );
    }

    __attribute__((always_inline))
    auto get_csr_nbrs() -> csr_graph_t& {
        return _csr_nbrs;
    }

    __attribute__((always_inline))
    auto get_csr_nbrs() const -> const csr_graph_t& {
        return _csr_nbrs;
    }

    __attribute__((always_inline))
    auto get_vecs_data() const -> const vector_array_t& {
        return _vecs_data;
    }

    /**
     * @brief Factory method to convert a FlatGraph to SearchGraph in parallel.
     * @param flat_graph The source flat graph to convert from.
     * @param fix_nbr_size Fixed number of neighbors per vertex in the search graph.
     * @return A new SearchGraph instance.
     */
    static auto from_flat_graph(
        const flat_graph_t& flat_graph,
        const vertex_num_t fix_nbr_size
    ) -> SearchGraph<IndexTraitsT> {
        const vertex_num_t num_vertices = flat_graph.get_num_vertices();
        const auto& vecs_data = flat_graph.get_vecs_data();
        const auto& nbrs_arr = flat_graph.get_nbrs_arr();

        // Create the search graph
        SearchGraph<IndexTraitsT> search_graph(num_vertices, fix_nbr_size, vecs_data);

        // Copy neighbors from FlatGraph to SearchGraph in parallel
        auto& csr_nbrs = search_graph.get_csr_nbrs();
        const vertex_id_t invalid_id = IndexTraitsT::invalid_vertex_id;

        tbb::parallel_for(
            tbb::blocked_range<vertex_id_t>(0, num_vertices),
            [&](const tbb::blocked_range<vertex_id_t>& r) {
                for (vertex_id_t vid = r.begin(); vid != r.end(); ++vid) {
                    const auto& nbrs = nbrs_arr[vid];
                    vertex_id_t* dst_nbrs = &csr_nbrs[static_cast<size_t>(vid) * fix_nbr_size];

                    // Copy up to fix_nbr_size neighbors
                    const vertex_num_t copy_count = std::min(
                        static_cast<vertex_num_t>(nbrs.size()),
                        fix_nbr_size
                    );

                    for (vertex_num_t i = 0; i < copy_count; ++i) {
                        dst_nbrs[i] = nbrs[i].get_id();
                    }

                    // Fill remaining slots with invalid vertex ID if needed
                    for (vertex_num_t i = copy_count; i < fix_nbr_size; ++i) {
                        dst_nbrs[i] = invalid_id;
                    }
                }
            }
        );

        return search_graph;
    }

private:
    /** @brief Number of vertices in the graph. */
    vertex_num_t _num_vertices;

    /** @brief Fixed number of neighbors per vertex. */
    vertex_num_t _fix_nbr_size;

    /** @brief CSR format neighbor storage: dense, cache-aligned array. */
    csr_graph_t _csr_nbrs;

    /** @brief Const reference to vector data for this graph. */
    const vector_array_t& _vecs_data;

};  // class SearchGraph

}   // namespace cpu
}   // namespace artea