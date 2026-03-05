/*
 * @FilePath: /Artea/include/artea/cpu/index/flat_graph.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Date: 2026-01-31
 * @Description: Flat graph structure for graph-based index.
 */

#pragma once

#include <cstddef>
#include <vector>
#include <string>
#include <fstream>
#include <limits>
#include <cstdint>
#include <type_traits>
#include <artea/common/logger.hpp>

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

    /**
     * @brief Snapshot flat graph to a binary file.
     * @param file_path Target file path.
     */
    auto snapshot(const std::string& file_path) const -> void {
        static_assert(
            std::is_trivially_copyable_v<vertex_id_t> && std::is_trivially_copyable_v<distance_t>,
            "vertex_id_t and distance_t must be trivially copyable for binary snapshot."
        );

        std::ofstream ofs(file_path, std::ios::binary | std::ios::trunc);
        if (!ofs.is_open()) {
            logger.error(fmt::format("Failed to open file for snapshotting flat graph: {}", file_path));
        }

        const uint32_t magic = k_file_magic;
        const uint32_t version = k_file_version;
        const vertex_num_t num_vertices = _num_vertices;
        const vertex_num_t reserved_nbr_size = _reserved_nbr_size;
        const vertex_num_t max_nbr_size = _max_nbr_size;

        ofs.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        ofs.write(reinterpret_cast<const char*>(&version), sizeof(version));
        ofs.write(reinterpret_cast<const char*>(&num_vertices), sizeof(num_vertices));
        ofs.write(reinterpret_cast<const char*>(&reserved_nbr_size), sizeof(reserved_nbr_size));
        ofs.write(reinterpret_cast<const char*>(&max_nbr_size), sizeof(max_nbr_size));

        // Write neighbor arrays
        for (vertex_num_t i = 0; i < num_vertices; ++i) {
            const auto& nbrs = _nbrs_arr[i];
            const vertex_num_t nbr_count = static_cast<vertex_num_t>(nbrs.size());
            ofs.write(reinterpret_cast<const char*>(&nbr_count), sizeof(nbr_count));

            if (nbr_count > 0) {
                ofs.write(
                    reinterpret_cast<const char*>(nbrs.data()),
                    static_cast<std::streamsize>(nbr_count * sizeof(nbr_t))
                );
            }
        }

        if (!ofs.good()) {
            logger.error(fmt::format("Failed while writing flat graph to file: {}", file_path));
        }
    }

    /**
     * @brief Restore flat graph from a snapshot binary file.
     * @param file_path Source file path.
     * @param vecs_data Reference to the vector data that this graph should bind to.
     * @return Loaded FlatGraph instance.
     */
    static auto restore(
        const std::string& file_path,
        const vector_array_t& vecs_data
    ) -> FlatGraph<IndexTraitsT> {
        static_assert(
            std::is_trivially_copyable_v<vertex_id_t> && std::is_trivially_copyable_v<distance_t>,
            "vertex_id_t and distance_t must be trivially copyable for binary loading."
        );

        std::ifstream ifs(file_path, std::ios::binary);
        if (!ifs.is_open()) {
            logger.error(fmt::format("Failed to open file for loading flat graph: {}", file_path));
        }

        uint32_t magic = 0;
        uint32_t version = 0;
        vertex_num_t num_vertices = 0;
        vertex_num_t reserved_nbr_size = 0;
        vertex_num_t max_nbr_size = 0;

        ifs.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        ifs.read(reinterpret_cast<char*>(&version), sizeof(version));
        ifs.read(reinterpret_cast<char*>(&num_vertices), sizeof(num_vertices));
        ifs.read(reinterpret_cast<char*>(&reserved_nbr_size), sizeof(reserved_nbr_size));
        ifs.read(reinterpret_cast<char*>(&max_nbr_size), sizeof(max_nbr_size));

        if (!ifs.good()) {
            logger.error(fmt::format("Failed to read flat graph header from file: {}", file_path));
        }

        if (magic != k_file_magic) {
            logger.error(fmt::format("Invalid flat graph file magic: {}", file_path));
        }

        if (version != k_file_version) {
            logger.error(fmt::format("Unsupported flat graph file version: {}", file_path));
        }

        FlatGraph<IndexTraitsT> flat_graph(vecs_data, num_vertices, max_nbr_size, reserved_nbr_size);

        // Read neighbor arrays
        for (vertex_num_t i = 0; i < num_vertices; ++i) {
            vertex_num_t nbr_count = 0;
            ifs.read(reinterpret_cast<char*>(&nbr_count), sizeof(nbr_count));

            if (nbr_count > 0) {
                flat_graph._nbrs_arr[i].resize(nbr_count);
                ifs.read(
                    reinterpret_cast<char*>(flat_graph._nbrs_arr[i].data()),
                    static_cast<std::streamsize>(nbr_count * sizeof(nbr_t))
                );
            }
        }

        if (!ifs.good()) {
            logger.error(fmt::format("Failed to read flat graph data from file: {}", file_path));
        }

        return flat_graph;
    }

protected:
    static constexpr uint32_t k_file_magic = 0x46474152;   // "FGRA"
    static constexpr uint32_t k_file_version = 1;

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

};  // class FlatGraph

}   // namespace cpu
}   // namespace artea