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
#include <string>
#include <fstream>
#include <limits>
#include <cstdint>
#include <type_traits>
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
     * @brief Snapshot search graph to a binary file.
     * @param file_path Target file path.
     */
    auto snapshot(const std::string& file_path) const -> void {
        static_assert(
            std::is_trivially_copyable_v<vertex_id_t>,
            "vertex_id_t must be trivially copyable for binary snapshot."
        );

        std::ofstream ofs(file_path, std::ios::binary | std::ios::trunc);
        if (!ofs.is_open()) {
            throw std::runtime_error("Failed to open file for snapshotting search graph: " + file_path);
        }

        const uint32_t magic = k_file_magic;
        const uint32_t version = k_file_version;
        const vertex_num_t num_vertices = _num_vertices;
        const vertex_num_t fix_nbr_size = _fix_nbr_size;
        const size_t csr_size = _csr_nbrs.size();

        ofs.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        ofs.write(reinterpret_cast<const char*>(&version), sizeof(version));
        ofs.write(reinterpret_cast<const char*>(&num_vertices), sizeof(num_vertices));
        ofs.write(reinterpret_cast<const char*>(&fix_nbr_size), sizeof(fix_nbr_size));
        ofs.write(reinterpret_cast<const char*>(&csr_size), sizeof(csr_size));

        if (!_csr_nbrs.empty()) {
            ofs.write(
                reinterpret_cast<const char*>(_csr_nbrs.data()),
                static_cast<std::streamsize>(_csr_nbrs.size() * sizeof(vertex_id_t))
            );
        }

        if (!ofs.good()) {
            throw std::runtime_error("Failed while writing search graph to file: " + file_path);
        }
    }

    /**
     * @brief Restore search graph from a snapshot binary file.
     * @param file_path Source file path.
     * @param vecs_data Reference to the vector data that this graph should bind to.
     * @return Loaded SearchGraph instance.
     */
    static auto restore(
        const std::string& file_path,
        const vector_array_t& vecs_data
    ) -> SearchGraph<IndexTraitsT> {
        static_assert(
            std::is_trivially_copyable_v<vertex_id_t>,
            "vertex_id_t must be trivially copyable for binary loading."
        );

        std::ifstream ifs(file_path, std::ios::binary);
        if (!ifs.is_open()) {
            throw std::runtime_error("Failed to open file for loading search graph: " + file_path);
        }

        uint32_t magic = 0;
        uint32_t version = 0;
        vertex_num_t num_vertices = 0;
        vertex_num_t fix_nbr_size = 0;
        size_t csr_size = 0;

        ifs.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        ifs.read(reinterpret_cast<char*>(&version), sizeof(version));
        ifs.read(reinterpret_cast<char*>(&num_vertices), sizeof(num_vertices));
        ifs.read(reinterpret_cast<char*>(&fix_nbr_size), sizeof(fix_nbr_size));
        ifs.read(reinterpret_cast<char*>(&csr_size), sizeof(csr_size));

        if (!ifs.good()) {
            throw std::runtime_error("Failed to read search graph header from file: " + file_path);
        }

        if (magic != k_file_magic) {
            throw std::runtime_error("Invalid search graph file magic: " + file_path);
        }

        if (version != k_file_version) {
            throw std::runtime_error("Unsupported search graph file version: " + file_path);
        }

        if (
            fix_nbr_size != 0
            && static_cast<size_t>(num_vertices)
                   > std::numeric_limits<size_t>::max() / static_cast<size_t>(fix_nbr_size)
        ) {
            throw std::runtime_error("CSR size multiplication overflows size_t: " + file_path);
        }

        const size_t expected_csr_size =
            static_cast<size_t>(num_vertices) * static_cast<size_t>(fix_nbr_size);
        if (csr_size != expected_csr_size) {
            throw std::runtime_error("Inconsistent CSR size in file: " + file_path);
        }

        SearchGraph<IndexTraitsT> search_graph(num_vertices, fix_nbr_size, vecs_data);
        if (search_graph._csr_nbrs.size() != csr_size) {
            throw std::runtime_error("Internal CSR size mismatch while loading: " + file_path);
        }

        if (!search_graph._csr_nbrs.empty()) {
            ifs.read(
                reinterpret_cast<char*>(search_graph._csr_nbrs.data()),
                static_cast<std::streamsize>(search_graph._csr_nbrs.size() * sizeof(vertex_id_t))
            );
        }

        if (!ifs.good()) {
            throw std::runtime_error("Failed to read search graph data from file: " + file_path);
        }

        return search_graph;
    }


private:
    static constexpr uint32_t k_file_magic = 0x41524753;   // "SGRA"
    static constexpr uint32_t k_file_version = 1;

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
