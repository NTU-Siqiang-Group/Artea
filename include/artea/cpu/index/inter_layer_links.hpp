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
 * @FilePath: /Artea/include/artea/cpu/index/inter_layer_links.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Date: 2026-03-05
 * @Description: Inter-layer links for hierarchical graph structures.
 */

#pragma once

#include <cstddef>
#include <vector>
#include <span>
#include <artea/common/logger.hpp>

namespace artea {
namespace cpu {

/**
 * @brief Inter-layer links for hierarchical graph structures.
 *
 * @details Per-layer vector storage:
 * - Layer 0 (bottom layer) does not need inter-layer links (it's the base layer)
 * - Layer i (i >= 1) stores links to its parent layer (Layer i-1)
 * - Each link is a vertex_id pointing to a vertex in the parent layer
 *
 * Storage Layout:
 * - _layer_links: Vector of vectors, one per layer (excluding Layer 0)
 *   - _layer_links[0] = Layer 1's links to Layer 0
 *   - _layer_links[i-1] = Layer i's links to Layer i-1
 *
 * Example with 3 layers (Layer 0, 1, 2):
 *   Layer 1 has 1000 vertices → 1000 links to Layer 0
 *   Layer 2 has 100 vertices  → 100 links to Layer 1
 *
 *   _layer_links[0] = [Layer1_links (1000 elements)]
 *   _layer_links[1] = [Layer2_links (100 elements)]
 *
 *   get_layer_links(1) returns _layer_links[0]  (Layer 1's links)
 *   get_layer_links(2) returns _layer_links[1]  (Layer 2's links)
 */
template <typename IndexTraitsT>
class InterLayerLinks {
    using vertex_num_t = typename IndexTraitsT::vertex_num_t;
    using vertex_id_t = typename IndexTraitsT::vertex_id_t;
    using layer_id_t = typename IndexTraitsT::layer_id_t;
    using layer_num_t = typename IndexTraitsT::layer_num_t;
    static constexpr layer_num_t expected_max_layers = 63;  // Arbitrary upper bound on number of layers for pre-allocation

public:
    explicit InterLayerLinks(const vertex_num_t num_vertices) {
        _layer_links.reserve(expected_max_layers);
    }

    /**
     * @brief Get inter-layer links for a specific layer.
     * @param layer_id The layer ID (must be >= 1, since Layer 0 has no links)
     * @return A span of vertex IDs pointing to the parent layer (Layer layer_id-1)
     */
    __attribute__((always_inline))
    auto get_layer_links(const layer_id_t layer_id) const -> std::span<const vertex_id_t> {
        return std::span<const vertex_id_t>(_layer_links[layer_id - 1]);
    }

    __attribute__((always_inline))
    auto get_layer_links(const layer_id_t layer_id) -> std::vector<vertex_id_t>& {
        return _layer_links[layer_id - 1];
    }

    /**
     * @brief Get the number of inter-layer links for a specific layer.
     * @param layer_id The layer ID (must be >= 1, since Layer 0 has no links)
     * @return The number of links for this layer
     */
    __attribute__((always_inline))
    auto get_num_layer_links(const layer_id_t layer_id) const -> vertex_num_t {
        return static_cast<vertex_num_t>(_layer_links[layer_id - 1].size());
    }

    /**
     * @brief Set inter-layer links for a specific layer.
     * @param layer_id The layer ID (must be >= 1)
     * @param links Container of vertex IDs pointing to the parent layer
     */
    template <typename ContainerT>
    __attribute__((always_inline))
    auto set_layer_links(const layer_id_t layer_id, ContainerT&& links) -> void {
        static_assert(std::is_same_v<typename std::decay_t<ContainerT>::value_type, vertex_id_t>,
                     "Container element type must match vertex_id_t");
        const auto idx = layer_id - 1;
        if (idx >= _layer_links.size()) {
            _layer_links.resize(idx + 1);
        }
        _layer_links[idx] = std::vector<vertex_id_t>(
            std::make_move_iterator(links.begin()),
            std::make_move_iterator(links.end()));
    }

    /**
     * @brief Append inter-layer links for a new layer from bottom to top.
     * @param links Container of vertex IDs pointing to the parent layer
     */
    template <typename ContainerT>
    __attribute__((always_inline))
    auto bottom_up_append(ContainerT&& links) -> void {
        static_assert(std::is_same_v<typename std::decay_t<ContainerT>::value_type, vertex_id_t>,
                     "Container element type must match vertex_id_t");
        _layer_links.emplace_back(
            std::make_move_iterator(links.begin()),
            std::make_move_iterator(links.end()));
    }

    __attribute__((always_inline))
    auto get_linked_vertex(const layer_id_t layer_id, const vertex_id_t vertex_id) const -> vertex_id_t {
        #ifndef NDEBUG
        if (layer_id - 1 >= _layer_links.size() ||
            vertex_id >= _layer_links[layer_id - 1].size()) {
            ARTEA_ERROR(fmt::format(
                "Invalid layer_id ({}) or vertex_id ({}) for inter-layer links",
                layer_id, vertex_id
            ));
        }
        #endif
        return _layer_links[layer_id - 1][vertex_id];
    }

private:
    /** @brief Per-layer vectors of inter-layer links (index i stores Layer i+1's links) */
    std::vector<std::vector<vertex_id_t>> _layer_links;

};  //  class InterLayerLinks

}   // namespace cpu
}   // namespace artea
