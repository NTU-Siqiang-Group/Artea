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
    using hierarchical_graph_t = typename IndexTraitsT::hierarchical_graph_t;
    using hierarchical_vecs_manager_t = typename IndexTraitsT::hierarchical_vecs_manager_t;
    using flat_graph_t = typename IndexTraitsT::flat_graph_t;
    using flat_graph_file_manager_t = typename IndexTraitsT::flat_graph_file_manager_t;
    using layer_config_t = typename IndexTraitsT::layer_config_t;
    using edges_builder_config_t = typename IndexTraitsT::artea_graph::edges_builder_config_t;
    using greedy_vertices_builder_config_t = typename IndexTraitsT::greedy_vertices_builder_config_t;
    using random_vertices_builder_config_t = typename IndexTraitsT::random_vertices_builder_config_t;
    using vertices_builder_config_t = typename IndexTraitsT::vertices_builder_config_t;
    using vector_array_t = typename IndexTraitsT::vector_array_t;

public:
    /**
     * @brief Snapshot hierarchical graph to a directory with metadata.
     * @param hierarchical_graph The hierarchical graph to snapshot.
     * @param index_dir Target directory path.
     * @param metadata Optional metadata to include in metadata.json.
     */
    static auto snapshot(
        const hierarchical_graph_t& hierarchical_graph,
        const std::string& index_dir,
        const nlohmann::json& metadata = nlohmann::json::object()
    ) -> void {
        // Create directory structure
        std::filesystem::create_directories(index_dir);
        std::filesystem::create_directories(index_dir + "/layers");

        // Write root metadata.json
        nlohmann::json meta = metadata;
        meta["graph_type"] = "hierarchical_graph";
        meta["version"] = "1.0";
        meta["num_vertices"] = hierarchical_graph.get_num_vertices();
        meta["num_layers"] = hierarchical_graph.get_num_layers();
        meta["bottom_layer_config"] = {
            {"max_nbr_size", hierarchical_graph.bottom_layer_config().max_nbr_size()},
            {"reserved_nbr_size", hierarchical_graph.bottom_layer_config().reserved_nbr_size()}
        };
        meta["upper_layer_config"] = {
            {"max_nbr_size", hierarchical_graph.upper_layer_config().max_nbr_size()},
            {"reserved_nbr_size", hierarchical_graph.upper_layer_config().reserved_nbr_size()}
        };
        meta["bottom_edges_builder_config"] = {
            {"scale_coeffs", hierarchical_graph.bottom_edges_builder_config().scale_coeffs()},
            {"shifted_coeffs", hierarchical_graph.bottom_edges_builder_config().shifted_coeffs()},
            {"num_outer_iters", hierarchical_graph.bottom_edges_builder_config().num_outer_iters()},
            {"num_inner_iters", hierarchical_graph.bottom_edges_builder_config().num_inner_iters()}
        };
        meta["upper_edges_builder_config"] = {
            {"scale_coeffs", hierarchical_graph.upper_edges_builder_config().scale_coeffs()},
            {"shifted_coeffs", hierarchical_graph.upper_edges_builder_config().shifted_coeffs()},
            {"num_outer_iters", hierarchical_graph.upper_edges_builder_config().num_outer_iters()},
            {"num_inner_iters", hierarchical_graph.upper_edges_builder_config().num_inner_iters()}
        };

        // Save vertices_builder_config based on which variant is active
        const auto& vertices_builder_config = hierarchical_graph.vertices_builder_config();
        if (std::holds_alternative<greedy_vertices_builder_config_t>(vertices_builder_config)) {
            const auto& config = std::get<greedy_vertices_builder_config_t>(vertices_builder_config);
            meta["vertices_builder_config"] = {
                {"type", "approx_rnet"},
                {"min_radius", config.min_radius()},
                {"beta", config.beta()},
                {"coverage_ratio", config.coverage_ratio()},
                {"confidence", config.confidence()},
                {"max_result_ratio", config.max_result_ratio()},
                {"sampling_batch_size", config.sampling_batch_size()}
            };
        } else if (std::holds_alternative<random_vertices_builder_config_t>(vertices_builder_config)) {
            const auto& config = std::get<random_vertices_builder_config_t>(vertices_builder_config);
            meta["vertices_builder_config"] = {
                {"type", "random"},
                {"random_result_ratio", config.random_result_ratio()}
            };
        }

        meta["entry_point"] = hierarchical_graph.get_entry_point();

        std::string metadata_path = index_dir + "/metadata.json";
        std::ofstream meta_ofs(metadata_path);
        if (!meta_ofs.is_open()) {
            logger.error(fmt::format("Failed to open metadata file: {}", metadata_path));
        }
        meta_ofs << meta.dump(2);
        meta_ofs.close();

        // Write inter_layer_links.bin
        std::string inter_layer_links_path = index_dir + "/inter_layer_links.bin";
        std::ofstream links_ofs(inter_layer_links_path, std::ios::binary);
        if (!links_ofs.is_open()) {
            logger.error(fmt::format("Failed to open inter_layer_links file: {}", inter_layer_links_path));
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
                logger.error(fmt::format("Failed to open layer metadata file: {}", layer_metadata_path));
            }
            layer_meta_ofs << layer_meta.dump(2);
            layer_meta_ofs.close();

            // Snapshot the flat graph using FlatGraphFileManager
            flat_graph_file_manager_t::snapshot(*layer_graphs[layer_id], layer_dir, layer_meta);
        }
    }

    /**
     * @brief Restore hierarchical graph from a snapshot directory.
     * @param index_dir Source directory path.
     * @param base_vecs Reference to the base layer vector data.
     * @return Loaded HierarchicalGraph instance.
     */
    static auto restore(
        const std::string& index_dir,
        const vector_array_t& base_vecs
    ) -> hierarchical_graph_t {
        // Read root metadata.json
        std::string metadata_path = index_dir + "/metadata.json";
        std::ifstream meta_ifs(metadata_path);
        if (!meta_ifs.is_open()) {
            logger.error(fmt::format("Failed to open metadata file: {}", metadata_path));
        }

        nlohmann::json meta;
        meta_ifs >> meta;
        meta_ifs.close();

        // Validate graph type
        if (meta["graph_type"] != "hierarchical_graph") {
            logger.error(fmt::format("Invalid graph type in metadata: {}", meta["graph_type"].get<std::string>()));
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

        // Extract edge builder configurations
        edges_builder_config_t bottom_edges_builder_config(
            meta["bottom_edges_builder_config"]["scale_coeffs"].get<typename edges_builder_config_t::ratio_t>(),
            meta["bottom_edges_builder_config"]["shifted_coeffs"].get<typename edges_builder_config_t::ratio_t>(),
            meta["bottom_edges_builder_config"]["num_outer_iters"].get<typename edges_builder_config_t::iter_t>(),
            meta["bottom_edges_builder_config"]["num_inner_iters"].get<typename edges_builder_config_t::iter_t>()
        );
        edges_builder_config_t upper_edges_builder_config(
            meta["upper_edges_builder_config"]["scale_coeffs"].get<typename edges_builder_config_t::ratio_t>(),
            meta["upper_edges_builder_config"]["shifted_coeffs"].get<typename edges_builder_config_t::ratio_t>(),
            meta["upper_edges_builder_config"]["num_outer_iters"].get<typename edges_builder_config_t::iter_t>(),
            meta["upper_edges_builder_config"]["num_inner_iters"].get<typename edges_builder_config_t::iter_t>()
        );

        // Restore vertices_builder_config based on type
        const std::string vertices_builder_config_type = meta["vertices_builder_config"]["type"].get<std::string>();
        vertices_builder_config_t vertices_builder_config = [&]() -> vertices_builder_config_t {
            if (vertices_builder_config_type == "approx_rnet") {
                return greedy_vertices_builder_config_t(
                    meta["vertices_builder_config"]["min_radius"].get<distance_t>(),
                    meta["vertices_builder_config"]["beta"].get<typename greedy_vertices_builder_config_t::ratio_t>(),
                    meta["vertices_builder_config"]["coverage_ratio"].get<typename greedy_vertices_builder_config_t::ratio_t>(),
                    meta["vertices_builder_config"]["confidence"].get<typename greedy_vertices_builder_config_t::ratio_t>(),
                    meta["vertices_builder_config"]["max_result_ratio"].get<typename greedy_vertices_builder_config_t::ratio_t>(),
                    meta["vertices_builder_config"]["sampling_batch_size"].get<vertex_num_t>()
                );
            } else if (vertices_builder_config_type == "random") {
                return random_vertices_builder_config_t(
                    meta["vertices_builder_config"]["random_result_ratio"].get<typename random_vertices_builder_config_t::ratio_t>()
                );
            }
            // This will throw and never return
            logger.error(fmt::format("Unknown vertices_builder_config type: {}", vertices_builder_config_type));
            throw std::runtime_error("Unreachable");  // Suppress compiler warning
        }();

        // Create hierarchical graph with base_vecs
        hierarchical_graph_t hier_graph(
            base_vecs,
            bottom_layer_config,
            upper_layer_config,
            bottom_edges_builder_config,
            upper_edges_builder_config,
            vertices_builder_config
        );

        const layer_num_t num_layers = meta["num_layers"].get<layer_num_t>();

        // Read inter_layer_links.bin
        std::string inter_layer_links_path = index_dir + "/inter_layer_links.bin";
        std::ifstream links_ifs(inter_layer_links_path, std::ios::binary);
        if (!links_ifs.is_open()) {
            logger.error(fmt::format("Failed to open inter_layer_links file: {}", inter_layer_links_path));
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
            auto layer_graph = flat_graph_file_manager_t::restore(layer_dir, layer_vecs);
            hier_graph.set_layer_graph(layer_id, std::move(layer_graph));
        }

        return hier_graph;
    }

};  // class HierarchicalGraphFileManager

}   // namespace cpu
}   // namespace artea