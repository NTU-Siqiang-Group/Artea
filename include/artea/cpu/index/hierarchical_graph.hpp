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
 * @FilePath: /Artea/include/artea/cpu/index/hierarchical_graph.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Date: 2026-03-05
 * @Description: Hierarchical graph structure for HNSW-like index.
 */

#pragma once

#include <cstddef>
#include <vector>
#include <memory>
#include <utility>
#include <variant>
#include <artea/common/logger.hpp>

namespace artea {
namespace cpu {

/**
 * @brief Hierarchical graph structure for HNSW-like index.
 * @tparam IndexTraitsT The index traits type.
 *
 * @note Layer ID mapping:
 *   - layer_id 0 is the bottom layer, stored at _layer_graphs[0]
 *   - Higher layer_id values represent upper layers
 *   - Direct mapping: _layer_graphs[layer_id]
 */
template <typename IndexTraitsT>
class HierarchicalGraph {

    using vertex_num_t = typename IndexTraitsT::vertex_num_t;
    using vertex_id_t = typename IndexTraitsT::vertex_id_t;
    using layer_id_t = typename IndexTraitsT::layer_id_t;
    using layer_num_t = typename IndexTraitsT::layer_num_t;
    using distance_t = typename IndexTraitsT::distance_t;
    using nbr_t = typename IndexTraitsT::nbr_t;
    using nbr_arr_t = typename IndexTraitsT::nbr_arr_t;
    using vector_array_t = typename IndexTraitsT::vector_array_t;
    using flat_graph_t = typename IndexTraitsT::flat_graph_t;
    using inter_layer_links_t = typename IndexTraitsT::inter_layer_links_t;
    using hierarchical_vecs_manager_t = typename IndexTraitsT::hierarchical_vecs_manager_t;
    using layer_config_t = typename IndexTraitsT::layer_config_t;
    using propagate_config_t = typename IndexTraitsT::artea_graph::propagate_config_t;
    using pruning_config_t = typename IndexTraitsT::artea_graph::pruning_config_t;
    using greedy_vertices_builder_config_t = typename IndexTraitsT::greedy_vertices_builder_config_t;
    using random_vertices_builder_config_t = typename IndexTraitsT::random_vertices_builder_config_t;
    using vertices_builder_config_t = typename IndexTraitsT::vertices_builder_config_t;

public:
    /**
     * @brief Construct a new Hierarchical Graph object.
     * @param base_vecs Reference to the base layer vector data.
     * @param bottom_layer_config Configuration for bottom layer.
     * @param upper_layer_config Configuration for upper layers.
     * @param bottom_pruning_config Pruning configuration for bottom layer.
     * @param upper_pruning_config Pruning configuration for upper layers.
     * @param propagate_config Propagation configuration (shared between layers).
     * @param vertices_builder_config Configuration for vertices builder (greedy or random).
     */
    HierarchicalGraph(
        const vector_array_t& base_vecs,
        const layer_config_t bottom_layer_config,
        const layer_config_t upper_layer_config,
        const pruning_config_t bottom_pruning_config,
        const pruning_config_t upper_pruning_config,
        const propagate_config_t propagate_config,
        const vertices_builder_config_t vertices_builder_config
    ) : _num_vertices(base_vecs.get_num_vecs()),
        _bottom_layer_config(bottom_layer_config),
        _upper_layer_config(upper_layer_config),
        _bottom_pruning_config(bottom_pruning_config),
        _upper_pruning_config(upper_pruning_config),
        _propagate_config(propagate_config),
        _vertices_builder_config(vertices_builder_config),
        _hier_vecs_manager(base_vecs),
        _inter_layer_links(_num_vertices)
    {
        // Note: _layer_graphs will be resized in HierarchicalEdgesBuilder after vertices construction
    }

    // Copying is deleted
    HierarchicalGraph(const HierarchicalGraph&) = delete;
    HierarchicalGraph& operator=(const HierarchicalGraph&) = delete;

    // Default move constructor and assignment
    HierarchicalGraph(HierarchicalGraph&&) noexcept = default;
    HierarchicalGraph& operator=(HierarchicalGraph&&) noexcept = default;

    // --- Public Interface ---

    __attribute__((always_inline))
    auto get_num_vertices() const -> vertex_num_t {
        return _num_vertices;
    }

    __attribute__((always_inline))
    auto get_num_layers() const -> layer_num_t {
        return _hier_vecs_manager.get_num_layers();
    }

    /**
     * @brief Resize the layer graphs vector to accommodate a specific number of layers.
     * @param num_layers The total number of layers (including bottom layer).
     */
    __attribute__((always_inline))
    auto resize(const layer_id_t num_layers) -> void {
        _layer_graphs.resize(num_layers);
    }

    __attribute__((always_inline))
    auto get_layer_graphs() -> std::vector<std::unique_ptr<flat_graph_t>>& {
        return _layer_graphs;
    }

    __attribute__((always_inline))
    auto get_layer_graphs() const -> const std::vector<std::unique_ptr<flat_graph_t>>& {
        return _layer_graphs;
    }

    __attribute__((always_inline))
    auto bottom_layer_config() const -> const layer_config_t& {
        return _bottom_layer_config;
    }

    __attribute__((always_inline))
    auto bottom_layer_config() -> layer_config_t& {
        return _bottom_layer_config;
    }

    __attribute__((always_inline))
    auto upper_layer_config() const -> const layer_config_t& {
        return _upper_layer_config;
    }

    __attribute__((always_inline))
    auto upper_layer_config() -> layer_config_t& {
        return _upper_layer_config;
    }

    __attribute__((always_inline))
    auto bottom_pruning_config() const -> const pruning_config_t& {
        return _bottom_pruning_config;
    }

    __attribute__((always_inline))
    auto bottom_pruning_config() -> pruning_config_t& {
        return _bottom_pruning_config;
    }

    __attribute__((always_inline))
    auto upper_pruning_config() const -> const pruning_config_t& {
        return _upper_pruning_config;
    }

    __attribute__((always_inline))
    auto upper_pruning_config() -> pruning_config_t& {
        return _upper_pruning_config;
    }

    __attribute__((always_inline))
    auto propagate_config() const -> const propagate_config_t& {
        return _propagate_config;
    }

    __attribute__((always_inline))
    auto propagate_config() -> propagate_config_t& {
        return _propagate_config;
    }

    __attribute__((always_inline))
    auto vertices_builder_config() const -> const vertices_builder_config_t& {
        return _vertices_builder_config;
    }

    __attribute__((always_inline))
    auto vertices_builder_config() -> vertices_builder_config_t& {
        return _vertices_builder_config;
    }

    __attribute__((always_inline))
    auto get_bottom_layer_graph() -> flat_graph_t& {
        return *_layer_graphs[0];
    }

    __attribute__((always_inline))
    auto get_bottom_layer_graph() const -> const flat_graph_t& {
        return *_layer_graphs[0];
    }

    /**
     * @brief Get the flat graph at a given layer_id.
     * @param layer_id The layer ID (0 for bottom layer, higher values for upper layers).
     */
    __attribute__((always_inline))
    auto get_layer_graph(const layer_id_t layer_id) -> flat_graph_t& {
        return *_layer_graphs[layer_id];
    }

    /**
     * @brief Get the flat graph at a given layer_id (const version).
     * @param layer_id The layer ID (0 for bottom layer, higher values for upper layers).
     */
    __attribute__((always_inline))
    auto get_layer_graph(const layer_id_t layer_id) const -> const flat_graph_t& {
        return *_layer_graphs[layer_id];
    }

    /**
     * @brief Set the flat graph at a given layer_id.
     * @param layer_id The layer ID (0 for bottom layer, higher values for upper layers).
     * @param layer_graph Unique pointer to the flat graph to set.
     */
    __attribute__((always_inline))
    auto set_layer_graph(const layer_id_t layer_id, std::unique_ptr<flat_graph_t> layer_graph) -> void {
        _layer_graphs[layer_id] = std::move(layer_graph);
    }

    /**
     * @brief Set the flat graph at a given layer_id (rvalue reference version).
     * @param layer_id The layer ID (0 for bottom layer, higher values for upper layers).
     * @param layer_graph Reference to the flat graph to set (will be moved).
     */
    __attribute__((always_inline))
    auto set_layer_graph(const layer_id_t layer_id, flat_graph_t&& layer_graph) -> void {
        _layer_graphs[layer_id] = std::make_unique<flat_graph_t>(std::move(layer_graph));
    }

    __attribute__((always_inline))
    auto get_vecs_data() const -> const vector_array_t& {
        return _hier_vecs_manager.get_base_vecs();
    }

    __attribute__((always_inline))
    auto get_base_vecs() const -> const vector_array_t& {
        return _hier_vecs_manager.get_base_vecs();
    }

    __attribute__((always_inline))
    auto get_hier_vecs_manager() const -> const hierarchical_vecs_manager_t& {
        return _hier_vecs_manager;
    }

    __attribute__((always_inline))
    auto get_hier_vecs_manager() -> hierarchical_vecs_manager_t& {
        return _hier_vecs_manager;
    }

    __attribute__((always_inline))
    auto get_inter_layer_links() -> inter_layer_links_t& {
        return _inter_layer_links;
    }

    __attribute__((always_inline))
    auto get_inter_layer_links() const -> const inter_layer_links_t& {
        return _inter_layer_links;
    }

    __attribute__((always_inline))
    auto get_entry_point() const -> vertex_id_t {
        return _entry_point;
    }

    __attribute__((always_inline))
    auto set_entry_point(const vertex_id_t entry_point) -> void {
        _entry_point = entry_point;
    }

protected:
    /** @brief Number of vertices in the graph. */
    vertex_num_t _num_vertices;

    /** @brief Configuration for bottom layer. */
    layer_config_t _bottom_layer_config;

    /** @brief Configuration for upper layers. */
    layer_config_t _upper_layer_config;

    /** @brief Pruning configuration for bottom layer. */
    pruning_config_t _bottom_pruning_config;

    /** @brief Pruning configuration for upper layers. */
    pruning_config_t _upper_pruning_config;

    /** @brief Propagation configuration (shared between layers). */
    propagate_config_t _propagate_config;

    /** @brief Vertices builder configuration (greedy or random). */
    vertices_builder_config_t _vertices_builder_config;

    /** @brief Hierarchical vector manager. */
    hierarchical_vecs_manager_t _hier_vecs_manager;

    /** @brief Flat graphs for all layers. layer_id 0 is the bottom layer at _layer_graphs[0]. */
    std::vector<std::unique_ptr<flat_graph_t>> _layer_graphs;

    inter_layer_links_t _inter_layer_links;

    /** @brief Entry point vertex ID for hierarchical search. */
    vertex_id_t _entry_point;

};  // class HierarchicalGraph

}   // namespace cpu
}   // namespace artea