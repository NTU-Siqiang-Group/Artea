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
 * @FilePath: /Artea/include/artea/cpu/index/hierarchical_search_graph.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Date: 2026-03-05
 * @Description: Hierarchical search graph with CSR format for HNSW-like index.
 */

#pragma once

#include <cstddef>
#include <vector>
#include <memory>

namespace artea {
namespace cpu {

/**
 * @brief Hierarchical search graph using CSR format for efficient HNSW-like queries.
 * @tparam IndexTraitsT The index traits type.
 *
 * @note Layer ID mapping:
 *   - layer_id 0 is the bottom layer, stored at _layer_graphs[0]
 *   - Higher layer_id values represent upper layers
 *   - Direct mapping: _layer_graphs[layer_id]
 */
template <typename IndexTraitsT>
class HierarchicalSearchGraph {

    using vertex_num_t = typename IndexTraitsT::vertex_num_t;
    using vertex_id_t = typename IndexTraitsT::vertex_id_t;
    using layer_id_t = typename IndexTraitsT::layer_id_t;
    using layer_num_t = typename IndexTraitsT::layer_num_t;
    using vector_array_t = typename IndexTraitsT::vector_array_t;
    using flat_search_graph_t = typename IndexTraitsT::flat_search_graph_t;
    using inter_layer_links_t = typename IndexTraitsT::inter_layer_links_t;
    using hierarchy_manager_t = typename IndexTraitsT::hierarchy_manager_t;

public:
    /**
     * @brief Construct a new Hierarchical Search Graph object.
     * @param hierarchy_manager Reference to the hierarchical vector manager.
     * @param bl_extracted_nbr_size Fixed number of neighbors for bottom layer.
     * @param ul_extracted_nbr_size Fixed number of neighbors for upper layers.
     */
    HierarchicalSearchGraph(
        const hierarchy_manager_t& hierarchy_manager,
        const vertex_num_t bl_extracted_nbr_size,
        const vertex_num_t ul_extracted_nbr_size
    ) : _num_vertices(hierarchy_manager.get_num_base_vecs()),
        _bl_extracted_nbr_size(bl_extracted_nbr_size),
        _ul_extracted_nbr_size(ul_extracted_nbr_size),
        _hierarchy_manager(hierarchy_manager),
        _inter_layer_links(inter_layer_links_t(_num_vertices))
    {}

    // Copying is deleted
    HierarchicalSearchGraph(const HierarchicalSearchGraph&) = delete;
    HierarchicalSearchGraph& operator=(const HierarchicalSearchGraph&) = delete;

    // Default move constructor and assignment
    HierarchicalSearchGraph(HierarchicalSearchGraph&&) noexcept = default;
    HierarchicalSearchGraph& operator=(HierarchicalSearchGraph&&) noexcept = default;

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
    __attribute__((always_inline))
    auto resize(const layer_id_t num_layers) -> void {
        _layer_graphs.resize(num_layers);
    }

    __attribute__((always_inline))
    auto get_bl_extracted_nbr_size() const -> vertex_num_t {
        return _bl_extracted_nbr_size;
    }

    __attribute__((always_inline))
    auto get_ul_extracted_nbr_size() const -> vertex_num_t {
        return _ul_extracted_nbr_size;
    }

    __attribute__((always_inline))
    auto get_bottom_layer_graph() -> flat_search_graph_t& {
        return *_layer_graphs[0];
    }

    __attribute__((always_inline))
    auto get_bottom_layer_graph() const -> const flat_search_graph_t& {
        return *_layer_graphs[0];
    }

    __attribute__((always_inline))
    auto get_layer_graphs() -> std::vector<std::unique_ptr<flat_search_graph_t>>& {
        return _layer_graphs;
    }

    __attribute__((always_inline))
    auto get_layer_graphs() const -> const std::vector<std::unique_ptr<flat_search_graph_t>>& {
        return _layer_graphs;
    }

    /**
     * @brief Get the flat search graph at a given layer_id.
     * @param layer_id The layer ID (0 for bottom layer, higher values for upper layers).
     */
    __attribute__((always_inline))
    auto get_layer_graph(const layer_id_t layer_id) -> flat_search_graph_t& {
        return *_layer_graphs[layer_id];
    }

    /**
     * @brief Get the flat search graph at a given layer_id (const version).
     * @param layer_id The layer ID (0 for bottom layer, higher values for upper layers).
     */
    __attribute__((always_inline))
    auto get_layer_graph(const layer_id_t layer_id) const -> const flat_search_graph_t& {
        return *_layer_graphs[layer_id];
    }

    /**
     * @brief Set the flat search graph at a given layer_id.
     * @param layer_id The layer ID (0 for bottom layer, higher values for upper layers).
     * @param layer_graph Unique pointer to the flat search graph to set.
     */
    __attribute__((always_inline))
    auto set_layer_graph(const layer_id_t layer_id, std::unique_ptr<flat_search_graph_t> layer_graph) -> void {
        _layer_graphs[layer_id] = std::move(layer_graph);
    }

    /**
     * @brief Set the flat search graph at a given layer_id (rvalue reference version).
     * @param layer_id The layer ID (0 for bottom layer, higher values for upper layers).
     * @param layer_graph Reference to the flat search graph to set (will be moved).
     */
    __attribute__((always_inline))
    auto set_layer_graph(const layer_id_t layer_id, flat_search_graph_t&& layer_graph) -> void {
        _layer_graphs[layer_id] = std::make_unique<flat_search_graph_t>(std::move(layer_graph));
    }

    __attribute__((always_inline))
    auto get_vecs_data() const -> const vector_array_t& {
        return _hierarchy_manager.get_base_vecs();
    }

    __attribute__((always_inline))
    auto get_base_vecs() const -> const vector_array_t& {
        return _hierarchy_manager.get_base_vecs();
    }

    __attribute__((always_inline))
    auto get_hierarchy_manager() const -> const hierarchy_manager_t& {
        return _hierarchy_manager;
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

    /** @brief Fixed number of neighbors for bottom layer. */
    vertex_num_t _bl_extracted_nbr_size;

    /** @brief Fixed number of neighbors for upper layers. */
    vertex_num_t _ul_extracted_nbr_size;

    /** @brief Hierarchical vector manager. */
    const hierarchy_manager_t& _hierarchy_manager;

    /** @brief Flat search graphs for all layers. layer_id 0 is the bottom layer at _layer_graphs[0]. */
    std::vector<std::unique_ptr<flat_search_graph_t>> _layer_graphs;

    /** @brief links vertex between two adjacent layers */
    inter_layer_links_t _inter_layer_links;

    /** @brief Entry point vertex ID for hierarchical search. */
    vertex_id_t _entry_point;

};  // class HierarchicalSearchGraph

}   // namespace cpu
}   // namespace artea
