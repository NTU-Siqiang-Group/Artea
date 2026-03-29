// Copyright 2026 Weitang Ye
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
 * @FilePath: /Artea/include/artea/cpu/persistence/flat_graph_file_manager.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Date: 2026-03-14
 * @Description: File manager for FlatGraph snapshot and restore operations.
 */

#pragma once

#include <string>
#include <fstream>
#include <filesystem>
#include <type_traits>
#include <nlohmann/json.hpp>
#include <artea/common/logger.hpp>

namespace artea {
namespace cpu {

/**
 * @brief File manager for FlatGraph snapshot and restore operations.
 * @tparam IndexTraitsT The index traits type.
 */
template <typename IndexTraitsT, typename FlatGraphT = typename IndexTraitsT::conv_graph_index_t>
class FlatGraphFileManager {

    using vertex_num_t = typename IndexTraitsT::vertex_num_t;
    using vertex_id_t = typename IndexTraitsT::vertex_id_t;
    using distance_t = typename IndexTraitsT::distance_t;
    using nbr_t = typename IndexTraitsT::nbr_t;
    using nbr_arr_t = typename IndexTraitsT::nbr_arr_t;
    using vector_array_t = typename IndexTraitsT::vector_array_t;
    using layer_config_t = typename IndexTraitsT::layer_config_t;

public:
    /**
     * @brief Snapshot flat graph to a directory with metadata.
     * @param flat_graph The flat graph to snapshot.
     * @param index_dir Target directory path.
     * @param metadata Optional metadata to include in metadata.json.
     */
    static auto snapshot(
        const FlatGraphT& flat_graph,
        const std::string& index_dir,
        const nlohmann::json& metadata = nlohmann::json::object()
    ) -> void {
        static_assert(
            std::is_trivially_copyable_v<vertex_id_t> && std::is_trivially_copyable_v<distance_t>,
            "vertex_id_t and distance_t must be trivially copyable for binary snapshot."
        );

        // Create directory if it doesn't exist
        std::filesystem::create_directories(index_dir);

        // Write metadata.json
        nlohmann::json meta = metadata;
        meta["graph_type"] = "flat_graph";
        meta["version"] = "1.0";
        meta["num_vertices"] = flat_graph.get_num_vertices();
        meta["layer_config"] = {
            {"max_nbr_size", flat_graph.layer_config().max_nbr_size()},
            {"reserved_nbr_size", flat_graph.layer_config().reserved_nbr_size()}
        };
        nlohmann::json subclass_meta = flat_graph.get_metadata();
        meta.merge_patch(subclass_meta);

        std::string metadata_path = index_dir + "/metadata.json";
        std::ofstream meta_ofs(metadata_path);
        if (!meta_ofs.is_open()) {
            ARTEA_ERROR(fmt::format("Failed to open metadata file: {}", metadata_path));
        }
        meta_ofs << meta.dump(2);
        meta_ofs.close();

        // Write binary graph data
        std::string graph_bin_path = index_dir + "/graph.bin";
        std::ofstream ofs(graph_bin_path, std::ios::binary | std::ios::trunc);
        if (!ofs.is_open()) {
            ARTEA_ERROR(fmt::format("Failed to open file for snapshotting flat graph: {}", graph_bin_path));
        }

        const uint32_t magic = k_file_magic;
        const uint32_t version = k_file_version;
        const vertex_num_t num_vertices = flat_graph.get_num_vertices();
        const vertex_num_t reserved_nbr_size = flat_graph.layer_config().reserved_nbr_size();
        const vertex_num_t max_nbr_size = flat_graph.layer_config().max_nbr_size();

        ofs.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        ofs.write(reinterpret_cast<const char*>(&version), sizeof(version));
        ofs.write(reinterpret_cast<const char*>(&num_vertices), sizeof(num_vertices));
        ofs.write(reinterpret_cast<const char*>(&reserved_nbr_size), sizeof(reserved_nbr_size));
        ofs.write(reinterpret_cast<const char*>(&max_nbr_size), sizeof(max_nbr_size));

        // Write neighbor arrays
        const auto& nbrs_arr = flat_graph.get_nbrs_arr();
        for (vertex_num_t i = 0; i < num_vertices; ++i) {
            const auto& nbrs = nbrs_arr[i];
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
            ARTEA_ERROR(fmt::format("Failed while writing flat graph to file: {}", graph_bin_path));
        }
    }

    /**
     * @brief Restore flat graph from a snapshot directory.
     * @param index_dir Source directory path.
     * @param vecs_data Reference to the vector data that this graph should bind to.
     * @return Loaded FlatGraph instance.
     */
    static auto restore(
        const std::string& index_dir,
        const vector_array_t& vecs_data
    ) -> FlatGraphT {
        static_assert(
            std::is_trivially_copyable_v<vertex_id_t> && std::is_trivially_copyable_v<distance_t>,
            "vertex_id_t and distance_t must be trivially copyable for binary loading."
        );

        // Read metadata.json
        std::string metadata_path = index_dir + "/metadata.json";
        std::ifstream meta_ifs(metadata_path);
        if (!meta_ifs.is_open()) {
            ARTEA_ERROR(fmt::format("Failed to open metadata file: {}", metadata_path));
        }

        nlohmann::json meta;
        meta_ifs >> meta;
        meta_ifs.close();

        // Validate graph type
        if (meta["graph_type"] != "flat_graph") {
            ARTEA_ERROR(fmt::format("Invalid graph type in metadata: {}", meta["graph_type"].get<std::string>()));
        }

        // Read binary graph data
        std::string graph_bin_path = index_dir + "/graph.bin";
        std::ifstream ifs(graph_bin_path, std::ios::binary);
        if (!ifs.is_open()) {
            ARTEA_ERROR(fmt::format("Failed to open file for loading flat graph: {}", graph_bin_path));
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
            ARTEA_ERROR(fmt::format("Failed to read flat graph header from file: {}", graph_bin_path));
        }

        if (magic != k_file_magic) {
            ARTEA_ERROR(fmt::format("Invalid flat graph file magic: {}", graph_bin_path));
        }

        if (version != k_file_version) {
            ARTEA_ERROR(fmt::format("Unsupported flat graph file version: {}", graph_bin_path));
        }

        layer_config_t layer_config(max_nbr_size, reserved_nbr_size);

        // Construct flat graph from metadata using subclass hook
        FlatGraphT flat_graph = FlatGraphT::from_metadata(meta, vecs_data, layer_config);

        // Check if the number of vertices matches
        if (flat_graph.get_num_vertices() != num_vertices) {
            ARTEA_ERROR(fmt::format(
                "Vertex count mismatch: vecs_data has {} vertices but file has {} vertices",
                flat_graph.get_num_vertices(), num_vertices
            ));
        }

        // Read neighbor arrays
        auto& nbrs_arr = flat_graph.get_nbrs_arr();
        for (vertex_num_t i = 0; i < num_vertices; ++i) {
            vertex_num_t nbr_count = 0;
            ifs.read(reinterpret_cast<char*>(&nbr_count), sizeof(nbr_count));

            if (nbr_count > 0) {
                nbrs_arr[i].resize(nbr_count);
                ifs.read(
                    reinterpret_cast<char*>(nbrs_arr[i].data()),
                    static_cast<std::streamsize>(nbr_count * sizeof(nbr_t))
                );
            }
        }

        if (!ifs.good()) {
            ARTEA_ERROR(fmt::format("Failed to read flat graph data from file: {}", graph_bin_path));
        }

        return flat_graph;
    }

private:
    static constexpr uint32_t k_file_magic = 0x46474152;   // "FGRA"
    static constexpr uint32_t k_file_version = 1;

};  // class FlatGraphFileManager

}   // namespace cpu
}   // namespace artea