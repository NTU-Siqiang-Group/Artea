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
 */
template <typename IndexTraitsT>
class HierarchicalGraph {

    using vertex_num_t = typename IndexTraitsT::vertex_num_t;
    using vertex_id_t = typename IndexTraitsT::vertex_id_t;
    using distance_t = typename IndexTraitsT::distance_t;
    using nbr_t = typename IndexTraitsT::nbr_t;
    using nbr_arr_t = typename IndexTraitsT::nbr_arr_t;
    using vector_array_t = typename IndexTraitsT::vector_array_t;
    using flat_graph_t = typename IndexTraitsT::flat_graph_t;

public:
    /**
     * @brief Construct a new Hierarchical Graph object.
     * @param vecs_data Reference to the vector data.
     * @param num_vertices The total number of vertices in the graph.
     * @param bl_max_nbr_size Maximum number of neighbors for bottom layer (default: 16).
     * @param bl_reserved_nbr_size Reserved neighbor size for bottom layer (default: 32).
     * @param ul_max_nbr_size Maximum number of neighbors for upper layers (default: 24).
     * @param ul_reserved_nbr_size Reserved neighbor size for upper layers (default: 48).
     */
    HierarchicalGraph(
        const vector_array_t& vecs_data,
        const vertex_num_t num_vertices,
        const vertex_num_t bl_max_nbr_size = 16,
        const vertex_num_t bl_reserved_nbr_size = 32,
        const vertex_num_t ul_max_nbr_size = 24,
        const vertex_num_t ul_reserved_nbr_size = 48
    ) : _num_vertices(num_vertices),
        _bl_max_nbr_size(bl_max_nbr_size),
        _bl_reserved_nbr_size(bl_reserved_nbr_size),
        _ul_max_nbr_size(ul_max_nbr_size),
        _ul_reserved_nbr_size(ul_reserved_nbr_size),
        _vecs_data(vecs_data),
        _bottom_layer_graph(std::make_unique<flat_graph_t>(vecs_data, num_vertices, bl_max_nbr_size, bl_reserved_nbr_size))
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
    auto get_num_layers() const -> vertex_num_t {
        return static_cast<vertex_num_t>(_upper_layer_graphs.size()) + 1;
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
        if (_bottom_layer_graph) {
            _bottom_layer_graph->set_max_nbr_size(max_nbr_size);
        }
    }

    __attribute__((always_inline))
    auto set_ul_max_nbr_size(const vertex_num_t max_nbr_size) -> void {
        _ul_max_nbr_size = max_nbr_size;
        for (auto& layer : _upper_layer_graphs) {
            if (layer) {
                layer->set_max_nbr_size(max_nbr_size);
            }
        }
    }

    __attribute__((always_inline))
    auto get_bottom_layer_graph() -> flat_graph_t& {
        return *_bottom_layer_graph;
    }

    __attribute__((always_inline))
    auto get_bottom_layer_graph() const -> const flat_graph_t& {
        return *_bottom_layer_graph;
    }

    __attribute__((always_inline))
    auto get_upper_layer_graphs() -> std::vector<std::unique_ptr<flat_graph_t>>& {
        return _upper_layer_graphs;
    }

    __attribute__((always_inline))
    auto get_upper_layer_graphs() const -> const std::vector<std::unique_ptr<flat_graph_t>>& {
        return _upper_layer_graphs;
    }

    __attribute__((always_inline))
    auto get_layer_graph(const vertex_num_t layer_id) -> flat_graph_t& {
        if (layer_id == 0) {
            return *_bottom_layer_graph;
        }
        return *_upper_layer_graphs[layer_id - 1];
    }

    __attribute__((always_inline))
    auto get_layer_graph(const vertex_num_t layer_id) const -> const flat_graph_t& {
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
     * @param layer_graph Unique pointer to the flat graph to set as bottom layer.
     */
    auto set_bottom_layer(std::unique_ptr<flat_graph_t> layer_graph) -> void {
        _bottom_layer_graph = std::move(layer_graph);
    }

    /**
     * @brief Set the bottom layer graph.
     * @param layer_graph Reference to the flat graph to set as bottom layer (will be moved).
     */
    auto set_bottom_layer(flat_graph_t&& layer_graph) -> void {
        _bottom_layer_graph = std::make_unique<flat_graph_t>(std::move(layer_graph));
    }

    /**
     * @brief Add an existing flat graph as a new upper layer.
     * @param layer_graph Unique pointer to the flat graph to add.
     */
    auto add_upper_layer(std::unique_ptr<flat_graph_t> layer_graph) -> void {
        _upper_layer_graphs.push_back(std::move(layer_graph));
    }

    /**
     * @brief Add an existing flat graph as a new upper layer.
     * @param layer_graph Reference to the flat graph to add (will be moved).
     */
    auto add_upper_layer(flat_graph_t&& layer_graph) -> void {
        _upper_layer_graphs.push_back(std::make_unique<flat_graph_t>(std::move(layer_graph)));
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

    /** @brief Const reference to vector data. */
    const vector_array_t& _vecs_data;

    /** @brief Flat graph for the bottom layer. */
    std::unique_ptr<flat_graph_t> _bottom_layer_graph;

    /** @brief Flat graphs for the upper layers. */
    std::vector<std::unique_ptr<flat_graph_t>> _upper_layer_graphs;

};  // class HierarchicalGraph

}   // namespace cpu
}   // namespace artea