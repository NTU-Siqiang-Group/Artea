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
 */
template <typename IndexTraitsT>
class HierarchicalSearchGraph {

    using vertex_num_t = typename IndexTraitsT::vertex_num_t;
    using vertex_id_t = typename IndexTraitsT::vertex_id_t;
    using vector_array_t = typename IndexTraitsT::vector_array_t;
    using flat_search_graph_t = typename IndexTraitsT::flat_search_graph_t;
    using inter_layer_links_t = typename IndexTraitsT::inter_layer_links_t;

public:
    /**
     * @brief Construct a new Hierarchical Search Graph object.
     * @param vecs_data Reference to the vector data.
     * @param num_vertices The total number of vertices in the graph.
     * @param bl_extracted_nbr_size Fixed number of neighbors for bottom layer.
     * @param ul_extracted_nbr_size Fixed number of neighbors for upper layers.
     */
    HierarchicalSearchGraph(
        const vector_array_t& vecs_data,
        const vertex_num_t num_vertices,
        const vertex_num_t bl_extracted_nbr_size,
        const vertex_num_t ul_extracted_nbr_size
    ) : _num_vertices(num_vertices),
        _bl_extracted_nbr_size(bl_extracted_nbr_size),
        _ul_extracted_nbr_size(ul_extracted_nbr_size),
        _vecs_data(vecs_data),
        _bottom_layer_graph(std::make_unique<flat_search_graph_t>(num_vertices, bl_extracted_nbr_size, vecs_data))
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
    auto get_num_layers() const -> vertex_num_t {
        return static_cast<vertex_num_t>(_upper_layer_graphs.size()) + 1;
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
        return *_bottom_layer_graph;
    }

    __attribute__((always_inline))
    auto get_bottom_layer_graph() const -> const flat_search_graph_t& {
        return *_bottom_layer_graph;
    }

    __attribute__((always_inline))
    auto get_upper_layer_graphs() -> std::vector<std::unique_ptr<flat_search_graph_t>>& {
        return _upper_layer_graphs;
    }

    __attribute__((always_inline))
    auto get_upper_layer_graphs() const -> const std::vector<std::unique_ptr<flat_search_graph_t>>& {
        return _upper_layer_graphs;
    }

    __attribute__((always_inline))
    auto get_layer_graph(const vertex_num_t layer_id) -> flat_search_graph_t& {
        if (layer_id == 0) {
            return *_bottom_layer_graph;
        }
        return *_upper_layer_graphs[layer_id - 1];
    }

    __attribute__((always_inline))
    auto get_layer_graph(const vertex_num_t layer_id) const -> const flat_search_graph_t& {
        if (layer_id == 0) {
            return *_bottom_layer_graph;
        }
        return *_upper_layer_graphs[layer_id - 1];
    }

    __attribute__((always_inline))
    auto get_vecs_data() const -> const vector_array_t& {
        return _vecs_data;
    }

    /**
     * @brief Set the bottom layer graph.
     * @param layer_graph Unique pointer to the flat search graph to set as bottom layer.
     */
    __attribute__((always_inline))
    auto set_bottom_layer(std::unique_ptr<flat_search_graph_t> layer_graph) -> void {
        _bottom_layer_graph = std::move(layer_graph);
    }

    /**
     * @brief Set the bottom layer graph.
     * @param layer_graph Reference to the flat search graph to set as bottom layer (will be moved).
     */
    __attribute__((always_inline))
    auto set_bottom_layer(flat_search_graph_t&& layer_graph) -> void {
        _bottom_layer_graph = std::make_unique<flat_search_graph_t>(std::move(layer_graph));
    }

    /**
     * @brief Add an existing flat search graph as a new upper layer.
     * @param layer_graph Unique pointer to the flat search graph to add.
     */
    __attribute__((always_inline))
    auto add_upper_layer(std::unique_ptr<flat_search_graph_t> layer_graph) -> void {
        _upper_layer_graphs.push_back(std::move(layer_graph));
    }

    /**
     * @brief Add an existing flat search graph as a new upper layer.
     * @param layer_graph Reference to the flat search graph to add (will be moved).
     */
    __attribute__((always_inline))
    auto add_upper_layer(flat_search_graph_t&& layer_graph) -> void {
        _upper_layer_graphs.push_back(std::make_unique<flat_search_graph_t>(std::move(layer_graph)));
    }

protected:
    /** @brief Number of vertices in the graph. */
    vertex_num_t _num_vertices;

    /** @brief Fixed number of neighbors for bottom layer. */
    vertex_num_t _bl_extracted_nbr_size;

    /** @brief Fixed number of neighbors for upper layers. */
    vertex_num_t _ul_extracted_nbr_size;

    /** @brief Const reference to vector data. */
    const vector_array_t& _vecs_data;

    /** @brief Flat search graph for the bottom layer. */
    std::unique_ptr<flat_search_graph_t> _bottom_layer_graph;

    /** @brief Flat search graphs for the upper layers. */
    std::vector<std::unique_ptr<flat_search_graph_t>> _upper_layer_graphs;

    /** @brief links vertex between two adjacent layers */
    inter_layer_links_t _inter_layer_links;

};  // class HierarchicalSearchGraph

}   // namespace cpu
}   // namespace artea
