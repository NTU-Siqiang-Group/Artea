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
 * @FilePath: /Artea/include/artea/cpu/index/compact_structure/hierarchical_graph.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Compact hierarchical graph: read-only multi-layer structure
 *               where each layer is a compact::InternalGraph (fixed-stride
 *               CSR with sentinel-terminated neighbors and inter-layer links).
 */

#pragma once

#include <cstddef>
#include <memory>
#include <vector>

namespace artea {
namespace cpu {
namespace compact {

/**
 * @brief Read-only hierarchical graph composed of compact::InternalGraph layers.
 *
 * Mirrors the API of dynamic::HierarchicalGraph but stores each layer as an
 * owned compact::InternalGraph (movable, no atomics/mutexes). Intended as the
 * query-time representation produced by HierarchicalGraphCompactor.
 *
 * Layers are stored as @c std::unique_ptr<internal_graph_t> in a dense vector
 * sized at construction. All layers must be installed via @c set_layer_graph
 * before the graph is used for queries.
 *
 * @tparam IndexTraitsT The index traits type.
 */
template <typename IndexTraitsT>
class HierarchicalGraph {

    using vertex_num_t     = typename IndexTraitsT::vertex_num_t;
    using vertex_id_t      = typename IndexTraitsT::vertex_id_t;
    using layer_num_t      = typename IndexTraitsT::layer_num_t;
    using layer_id_t       = typename IndexTraitsT::layer_id_t;
    using internal_graph_t = typename IndexTraitsT::compact::internal_graph_t;

public:
    /**
     * @brief Construct a compact HierarchicalGraph with @p num_layers slots.
     * @param num_layers Number of layers (all initially null).
     */
    explicit HierarchicalGraph(const layer_num_t num_layers)
        : _layer_graphs(num_layers), _num_layers(num_layers) {}

    HierarchicalGraph(const HierarchicalGraph&) = delete;
    HierarchicalGraph& operator=(const HierarchicalGraph&) = delete;

    HierarchicalGraph(HierarchicalGraph&&) noexcept = default;
    HierarchicalGraph& operator=(HierarchicalGraph&&) noexcept = default;

    ~HierarchicalGraph() = default;

    // --- Layer Access ---

    __attribute__((always_inline))
    auto get_num_layers() const -> layer_num_t { return _num_layers; }

    __attribute__((always_inline))
    auto get_layer_graph(const layer_id_t layer_id) -> internal_graph_t& {
        return *_layer_graphs[layer_id];
    }

    __attribute__((always_inline))
    auto get_layer_graph(const layer_id_t layer_id) const -> const internal_graph_t& {
        return *_layer_graphs[layer_id];
    }

    /**
     * @brief Install a layer graph at the given slot (transfers ownership).
     */
    __attribute__((always_inline))
    auto set_layer_graph(const layer_id_t layer_id,
                         std::unique_ptr<internal_graph_t> graph) -> void {
        _layer_graphs[layer_id] = std::move(graph);
    }

    __attribute__((always_inline))
    auto get_layer_graphs() -> std::vector<std::unique_ptr<internal_graph_t>>& {
        return _layer_graphs;
    }

    __attribute__((always_inline))
    auto get_layer_graphs() const
        -> const std::vector<std::unique_ptr<internal_graph_t>>& {
        return _layer_graphs;
    }

private:
    std::vector<std::unique_ptr<internal_graph_t>> _layer_graphs;
    layer_num_t _num_layers;

};  // class HierarchicalGraph

}   // namespace compact
}   // namespace cpu
}   // namespace artea
