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

public:
    /**
     * @brief Construct a new Hierarchical Graph object.
     * @param hier_vecs_manager Reference to the hierarchical vector manager.
     * @param num_vertices The total number of vertices in the graph.
     * @param bl_max_nbr_size Maximum number of neighbors for bottom layer (default: 16).
     * @param bl_reserved_nbr_size Reserved neighbor size for bottom layer (default: 32).
     * @param ul_max_nbr_size Maximum number of neighbors for upper layers (default: 24).
     * @param ul_reserved_nbr_size Reserved neighbor size for upper layers (default: 48).
     */
    HierarchicalGraph(
        const hierarchical_vecs_manager_t& hier_vecs_manager,
        const vertex_num_t num_vertices,
        const vertex_num_t bl_max_nbr_size = 16,
        const vertex_num_t ul_max_nbr_size = 24,
        const vertex_num_t bl_reserved_nbr_size = 32,
        const vertex_num_t ul_reserved_nbr_size = 48
    ) : _num_vertices(num_vertices),
        _bl_max_nbr_size(bl_max_nbr_size),
        _ul_max_nbr_size(ul_max_nbr_size),
        _bl_reserved_nbr_size(bl_reserved_nbr_size),
        _ul_reserved_nbr_size(ul_reserved_nbr_size),
        _hier_vecs_manager(hier_vecs_manager),
        _inter_layer_links(inter_layer_links_t(num_vertices))
    {}

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
        return static_cast<layer_num_t>(_layer_graphs.size());
    }

    /**
     * @brief Resize the layer graphs vector to accommodate a specific number of layers.
     * @param num_layers The total number of layers (including bottom layer).
     */
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
    auto get_bl_max_nbr_size() const -> vertex_num_t {
        return _bl_max_nbr_size;
    }

    __attribute__((always_inline))
    auto get_bl_reserved_nbr_size() const -> vertex_num_t {
        return _bl_reserved_nbr_size;
    }

    __attribute__((always_inline))
    auto get_ul_max_nbr_size() const -> vertex_num_t {
        return _ul_max_nbr_size;
    }

    __attribute__((always_inline))
    auto get_ul_reserved_nbr_size() const -> vertex_num_t {
        return _ul_reserved_nbr_size;
    }

    __attribute__((always_inline))
    auto set_bl_max_nbr_size(const vertex_num_t max_nbr_size) -> void {
        _bl_max_nbr_size = max_nbr_size;
        if (!_layer_graphs.empty() && _layer_graphs[0]) {
            _layer_graphs[0]->set_max_nbr_size(max_nbr_size);
        }
    }

    __attribute__((always_inline))
    auto set_ul_max_nbr_size(const vertex_num_t max_nbr_size) -> void {
        _ul_max_nbr_size = max_nbr_size;
        for (size_t i = 1; i < _layer_graphs.size(); ++i) {
            if (_layer_graphs[i]) {
                _layer_graphs[i]->set_max_nbr_size(max_nbr_size);
            }
        }
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
    auto set_layer_graph(const layer_id_t layer_id, std::unique_ptr<flat_graph_t> layer_graph) -> void {
        _layer_graphs[layer_id] = std::move(layer_graph);
    }

    /**
     * @brief Set the flat graph at a given layer_id (rvalue reference version).
     * @param layer_id The layer ID (0 for bottom layer, higher values for upper layers).
     * @param layer_graph Reference to the flat graph to set (will be moved).
     */
    auto set_layer_graph(const layer_id_t layer_id, flat_graph_t&& layer_graph) -> void {
        _layer_graphs[layer_id] = std::make_unique<flat_graph_t>(std::move(layer_graph));
    }

    __attribute__((always_inline))
    auto get_vecs_data() const -> const vector_array_t& {
        return _hier_vecs_manager.get_base_vecs();
    }

    __attribute__((always_inline))
    auto get_hier_vecs_manager() const -> const hierarchical_vecs_manager_t& {
        return _hier_vecs_manager;
    }

protected:
    /** @brief Number of vertices in the graph. */
    vertex_num_t _num_vertices;

    /** @brief Maximum number of neighbors for bottom layer. */
    vertex_num_t _bl_max_nbr_size;

    /** @brief Reserved neighbor size for bottom layer. */
    vertex_num_t _bl_reserved_nbr_size;

    /** @brief Maximum number of neighbors for upper layers. */
    vertex_num_t _ul_max_nbr_size;

    /** @brief Reserved neighbor size for upper layers. */
    vertex_num_t _ul_reserved_nbr_size;

    /** @brief Hierarchical vector manager. */
    const hierarchical_vecs_manager_t& _hier_vecs_manager;

    /** @brief Flat graphs for all layers. layer_id 0 is the bottom layer at _layer_graphs[0]. */
    std::vector<std::unique_ptr<flat_graph_t>> _layer_graphs;

    inter_layer_links_t _inter_layer_links;

};  // class HierarchicalGraph

}   // namespace cpu
}   // namespace artea