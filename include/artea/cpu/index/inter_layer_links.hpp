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
 * @brief Inter-layer links for hierarchical graph structures using CSR format.
 *
 * @details CSR (Compressed Sparse Row) Structure:
 * - Layer 0 (bottom layer) does not need inter-layer links (it's the base layer)
 * - Layer i (i >= 1) stores links to its parent layer (Layer i-1)
 * - Each link is a vertex_id pointing to a vertex in the parent layer
 *
 * Storage Layout:
 * - _links_arr: Flattened array storing all inter-layer links
 * - _layer_offsets: CSR offsets array (size = num_layers + 1)
 *   - _layer_offsets[0] = 0 (sentinel, start of Layer 1's links)
 *   - _layer_offsets[i] = end offset of Layer i's links (i >= 1)
 *
 * Example with 3 layers (Layer 0, 1, 2):
 *   Layer 1 has 1000 vertices → 1000 links to Layer 0
 *   Layer 2 has 100 vertices  → 100 links to Layer 1
 *
 *   _links_arr = [Layer1_links (1000 elements) | Layer2_links (100 elements)]
 *   _layer_offsets = [0, 1000, 1100]
 *
 *   get_layer_links(1) returns _links_arr[0..1000]     (Layer 1's links)
 *   get_layer_links(2) returns _links_arr[1000..1100]  (Layer 2's links)
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
        // expected number of total inter-layer links
        _links_arr.reserve(num_vertices * 2);
        _layer_offsets.reserve(expected_max_layers + 1);
        _layer_offsets.push_back(0);  // CSR sentinel: offset[0] = 0
    }

    /**
     * @brief Get inter-layer links for a specific layer.
     * @param layer_id The layer ID (must be >= 1, since Layer 0 has no links)
     * @return A span of vertex IDs pointing to the parent layer (Layer layer_id-1)
     *
     * @note CSR indexing: Layer i's links are stored in _links_arr[offset[i-1]..offset[i]]
     *       This is because Layer 0 doesn't need links, so Layer 1's links start at offset[0].
     */
    __attribute__((always_inline))
    auto get_layer_links(const layer_id_t layer_id) const -> std::span<const vertex_id_t> {
        return std::span<const vertex_id_t>(
            _links_arr.data() + _layer_offsets[layer_id - 1],
            _layer_offsets[layer_id] - _layer_offsets[layer_id - 1]
        );
    }

    /**
     * @brief Get the number of inter-layer links for a specific layer.
     * @param layer_id The layer ID (must be >= 1, since Layer 0 has no links)
     * @return The number of links for this layer
     */
    __attribute__((always_inline))
    auto get_num_layer_links(const layer_id_t layer_id) const -> vertex_num_t {
        return _layer_offsets[layer_id] - _layer_offsets[layer_id - 1];
    }

    /**
     * @brief Set inter-layer links for a specific layer.
     * @param layer_id The layer ID (must be >= 1)
     * @param links Container of vertex IDs pointing to the parent layer
     *
     * @note The layer_id parameter is provided for semantic clarity but not used in indexing.
     *       Links are appended to _links_arr in order (Layer 1, Layer 2, ...).
     *       The CSR offset for this layer is automatically recorded.
     */
    template <typename ContainerT>
    __attribute__((always_inline))
    auto set_layer_links(const layer_id_t layer_id, ContainerT&& links) -> void {
        static_assert(std::is_same_v<typename std::decay_t<ContainerT>::value_type, vertex_id_t>,
                     "Container element type must match vertex_id_t");
        _links_arr.insert(_links_arr.end(),
                         std::make_move_iterator(links.begin()),
                         std::make_move_iterator(links.end()));
        _layer_offsets.push_back(static_cast<vertex_num_t>(_links_arr.size()));
    }

    /**
     * @brief Append inter-layer links for a new layer from bottom to top.
     * @param links Container of vertex IDs pointing to the parent layer
     *
     * @note This is a convenience method that automatically tracks layer IDs.
     *       Links are appended to _links_arr in order (Layer 1, Layer 2, ...).
     *       The CSR offset for this layer is automatically recorded.
     */
    template <typename ContainerT>
    __attribute__((always_inline))
    auto bottom_up_append(ContainerT&& links) -> void {
        static_assert(std::is_same_v<typename std::decay_t<ContainerT>::value_type, vertex_id_t>,
                     "Container element type must match vertex_id_t");
        _links_arr.insert(_links_arr.end(),
                         std::make_move_iterator(links.begin()),
                         std::make_move_iterator(links.end()));
        _layer_offsets.push_back(static_cast<vertex_num_t>(_links_arr.size()));
    }

    __attribute__((always_inline))
    auto get_linked_vertex(const layer_id_t layer_id, const vertex_id_t vertex_id) const -> vertex_id_t {
        #ifndef NDEBUG
        if (layer_id >= _layer_offsets.size() ||
            vertex_id >= _layer_offsets[layer_id] - _layer_offsets[layer_id - 1]) {
            logger.error(fmt::format(
                "Invalid layer_id ({}) or vertex_id ({}) for inter-layer links",
                layer_id, vertex_id
            ));
        }
        #endif
        return _links_arr[_layer_offsets[layer_id - 1] + vertex_id];
    }

private:
    /** @brief Flattened array of inter-layer links */
    std::vector<vertex_id_t> _links_arr;

    /** @brief CSR offsets into _links_arr for each layer (size = num_layers + 1) */
    std::vector<vertex_num_t> _layer_offsets;

};  //  class InterLayerLinks

}   // namespace cpu
}   // namespace artea
