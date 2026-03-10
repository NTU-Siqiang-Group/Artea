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

    __attribute__((always_inline))
    auto get_layer_links(const layer_id_t layer_id) const -> std::span<const vertex_id_t> {
        return std::span<const vertex_id_t>(
            _links_arr.data() + _layer_offsets[layer_id],
            _layer_offsets[layer_id + 1] - _layer_offsets[layer_id]
        );
    }

    __attribute__((always_inline))
    auto add_layer_links(const layer_id_t layer_id, const std::vector<vertex_id_t>& links) -> void {
        _links_arr.insert(_links_arr.end(), links.begin(), links.end());
        _layer_offsets.push_back(static_cast<vertex_num_t>(_links_arr.size()));
    }

    __attribute__((always_inline))
    auto get_linked_point(const layer_id_t layer_id, const vertex_id_t vertex_id) const -> vertex_id_t {
        #ifndef NDEBUG
        if (layer_id >= _layer_offsets.size() - 1 ||
            vertex_id >= _layer_offsets[layer_id + 1] - _layer_offsets[layer_id]) {
            logger.error(fmt::format(
                "Invalid layer_id ({}) or vertex_id ({}) for inter-layer links",
                layer_id, vertex_id
            ));
        }
        #endif
        return _links_arr[_layer_offsets[layer_id] + vertex_id];
    }

private:
    /** @brief Flattened array of inter-layer links */
    std::vector<vertex_id_t> _links_arr;

    /** @brief CSR offsets into _links_arr for each layer (size = num_layers + 1) */
    std::vector<vertex_num_t> _layer_offsets;

};  //  class InterLayerLinks

}   // namespace cpu
}   // namespace artea
