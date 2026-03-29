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
 * @FilePath: /Artea/include/artea/cpu/persistence/hierarchical_graph_file_manager.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Date: 2026-03-14
 * @Description: File manager for HierarchicalGraph snapshot and restore operations.
 */

#pragma once

#include <string>
#include <fstream>
#include <filesystem>
#include <variant>
#include <nlohmann/json.hpp>
#include <artea/common/logger.hpp>

namespace artea {
namespace cpu {

/**
 * @brief File manager for HierarchicalGraph snapshot and restore operations.
 * @tparam IndexTraitsT The index traits type.
 */
template <typename IndexTraitsT>
class HierarchicalGraphFileManager {

    using vertex_num_t = typename IndexTraitsT::vertex_num_t;
    using vertex_id_t = typename IndexTraitsT::vertex_id_t;
    using layer_id_t = typename IndexTraitsT::layer_id_t;
    using layer_num_t = typename IndexTraitsT::layer_num_t;
    using distance_t = typename IndexTraitsT::distance_t;
    using hierarchical_vecs_manager_t = typename IndexTraitsT::hierarchical_vecs_manager_t;
    using layer_config_t = typename IndexTraitsT::layer_config_t;
    using vector_array_t = typename IndexTraitsT::vector_array_t;

public:
    /**
     * @brief Snapshot hierarchical graph to a directory with metadata.
     * @param hierarchical_graph The hierarchical graph to snapshot.
     * @param index_dir Target directory path.
     * @param metadata Optional metadata to include in metadata.json.
     */
    template <typename HierGraphT>
    static auto snapshot(
        const HierGraphT& hierarchical_graph,
        const std::string& index_dir,
        const nlohmann::json& metadata = nlohmann::json::object()
    ) -> void {
        // Create directory structure
        std::filesystem::create_directories(index_dir);
        std::filesystem::create_directories(index_dir + "/layers");

        // Write root metadata.json
        nlohmann::json meta = metadata;
        meta.merge_patch(hierarchical_graph.get_base_metadata());
        meta.merge_patch(hierarchical_graph.get_metadata());

        meta["entry_point"] = hierarchical_graph.get_entry_point();

        std::string metadata_path = index_dir + "/metadata.json";
        std::ofstream meta_ofs(metadata_path);
        if (!meta_ofs.is_open()) {
            ARTEA_ERROR(fmt::format("Failed to open metadata file: {}", metadata_path));
        }
        meta_ofs << meta.dump(2);
        meta_ofs.close();

        // Write inter_layer_links.bin
        std::string inter_layer_links_path = index_dir + "/inter_layer_links.bin";
        std::ofstream links_ofs(inter_layer_links_path, std::ios::binary);
        if (!links_ofs.is_open()) {
            ARTEA_ERROR(fmt::format("Failed to open inter_layer_links file: {}", inter_layer_links_path));
        }

        // Manually serialize inter_layer_links
        const auto& inter_layer_links = hierarchical_graph.get_inter_layer_links();
        const layer_num_t num_layers = hierarchical_graph.get_num_layers();

        // Write each layer's links
        for (layer_id_t layer_id = 1; layer_id < num_layers; ++layer_id) {
            const auto layer_links = inter_layer_links.get_layer_links(layer_id);
            const vertex_num_t num_links = static_cast<vertex_num_t>(layer_links.size());

            // Write number of links for this layer
            links_ofs.write(reinterpret_cast<const char*>(&num_links), sizeof(vertex_num_t));

            // Write the links data
            links_ofs.write(reinterpret_cast<const char*>(layer_links.data()),
                           static_cast<std::streamsize>(num_links * sizeof(vertex_id_t)));
        }
        links_ofs.close();

        // Write each layer
        const auto& layer_graphs = hierarchical_graph.get_layer_graphs();
        for (layer_id_t layer_id = 0; layer_id < num_layers; ++layer_id) {
            std::string layer_dir = index_dir + "/layers/layer_" + std::to_string(layer_id);
            std::filesystem::create_directories(layer_dir);

            // Write layer metadata
            nlohmann::json layer_meta;
            layer_meta["layer_id"] = layer_id;
            layer_meta["num_vertices"] = layer_graphs[layer_id]->get_num_vertices();
            layer_meta["layer_config"] = {
                {"max_nbr_size", layer_graphs[layer_id]->layer_config().max_nbr_size()},
                {"reserved_nbr_size", layer_graphs[layer_id]->layer_config().reserved_nbr_size()}
            };

            std::string layer_metadata_path = layer_dir + "/metadata.json";
            std::ofstream layer_meta_ofs(layer_metadata_path);
            if (!layer_meta_ofs.is_open()) {
                ARTEA_ERROR(fmt::format("Failed to open layer metadata file: {}", layer_metadata_path));
            }
            layer_meta_ofs << layer_meta.dump(2);
            layer_meta_ofs.close();

            // Snapshot the flat graph using FlatGraphFileManager
            FlatGraphFileManager<IndexTraitsT>::snapshot(*layer_graphs[layer_id], layer_dir, layer_meta);
        }
    }

    /**
     * @brief Restore hierarchical graph from a snapshot directory.
     * @param index_dir Source directory path.
     * @param base_vecs Reference to the base layer vector data.
     * @return Loaded HierarchicalGraph instance.
     */
    template <typename HierGraphT>
    static auto restore(
        const std::string& index_dir,
        const vector_array_t& base_vecs
    ) -> HierGraphT {
        // Read root metadata.json
        std::string metadata_path = index_dir + "/metadata.json";
        std::ifstream meta_ifs(metadata_path);
        if (!meta_ifs.is_open()) {
            ARTEA_ERROR(fmt::format("Failed to open metadata file: {}", metadata_path));
        }

        nlohmann::json meta;
        meta_ifs >> meta;
        meta_ifs.close();

        // Validate graph type
        if (meta["graph_type"] != "hierarchical_graph") {
            ARTEA_ERROR(fmt::format("Invalid graph type in metadata: {}", meta["graph_type"].get<std::string>()));
        }

        // Extract configuration
        layer_config_t bottom_layer_config(
            meta["bottom_layer_config"]["max_nbr_size"].get<vertex_num_t>(),
            meta["bottom_layer_config"]["reserved_nbr_size"].get<vertex_num_t>()
        );
        layer_config_t upper_layer_config(
            meta["upper_layer_config"]["max_nbr_size"].get<vertex_num_t>(),
            meta["upper_layer_config"]["reserved_nbr_size"].get<vertex_num_t>()
        );

        // Construct hierarchical graph from metadata using subclass hook
        HierGraphT hier_graph = HierGraphT::from_metadata(
            meta, base_vecs, bottom_layer_config, upper_layer_config
        );

        const layer_num_t num_layers = meta["num_layers"].get<layer_num_t>();

        // Read inter_layer_links.bin
        std::string inter_layer_links_path = index_dir + "/inter_layer_links.bin";
        std::ifstream links_ifs(inter_layer_links_path, std::ios::binary);
        if (!links_ifs.is_open()) {
            ARTEA_ERROR(fmt::format("Failed to open inter_layer_links file: {}", inter_layer_links_path));
        }

        // Get references to hier_vecs_manager and inter_layer_links
        auto& hier_vecs_manager = hier_graph.get_hier_vecs_manager();
        auto& inter_layer_links = hier_graph.get_inter_layer_links();

        // Reconstruct layers from inter_layer_links (Layer 1 to num_layers-1)
        for (layer_id_t layer_id = 1; layer_id < num_layers; ++layer_id) {
            vertex_num_t num_links;
            links_ifs.read(reinterpret_cast<char*>(&num_links), sizeof(vertex_num_t));

            std::vector<vertex_id_t> layer_links(num_links);
            links_ifs.read(reinterpret_cast<char*>(layer_links.data()),
                          static_cast<std::streamsize>(num_links * sizeof(vertex_id_t)));

            // Extract subset from parent layer
            const auto& parent_layer_vecs = hier_vecs_manager.get_layer_vecs(layer_id - 1);
            vector_array_t layer_vecs = parent_layer_vecs.extract_subset(layer_links);

            // Append to hier_vecs_manager and inter_layer_links
            inter_layer_links.bottom_up_append(std::move(layer_links));
            hier_vecs_manager.bottom_up_append(std::move(layer_vecs));
        }
        links_ifs.close();

        // Read entry_point from metadata
        hier_graph.set_entry_point(meta["entry_point"].get<vertex_id_t>());

        // Resize layer_graphs to match the number of layers
        hier_graph.resize(num_layers);

        // Load each layer
        for (layer_id_t layer_id = 0; layer_id < num_layers; ++layer_id) {
            std::string layer_dir = index_dir + "/layers/layer_" + std::to_string(layer_id);

            // Get the appropriate vector data for this layer
            const auto& layer_vecs = hier_graph.get_hier_vecs_manager().get_layer_vecs(layer_id);

            // Restore the flat graph using FlatGraphFileManager
            using layer_graph_t = typename HierGraphT::layer_graph_t;
            auto layer_graph = FlatGraphFileManager<IndexTraitsT>::template restore<layer_graph_t>(layer_dir, layer_vecs);
            hier_graph.set_layer_graph(layer_id, std::move(layer_graph));
        }

        return hier_graph;
    }

};  // class HierarchicalGraphFileManager

}   // namespace cpu
}   // namespace artea