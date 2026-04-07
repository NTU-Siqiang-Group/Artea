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
 * @FilePath: /Artea/include/artea/cpu/index/layer_nbr.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Layer Neighbor structure definition.
 */

#pragma once

#include <cstdint>
#include <type_traits>
#include <limits>

namespace artea {
namespace cpu {

/**
 * @brief Layer neighbor structure storing dual vertex identifiers.
 *
 * Each LayerNeighbor records two identifiers for a neighbor vertex:
 * - @p base_vid : the position of this neighbor's vector in the global vector table.
 * - @p layer_vid : the position of this neighbor's adjacency list in the layer-local storage.
 */
template <typename BaseTraitsT>
struct LayerNeighbor {

    using vertex_id_t = typename BaseTraitsT::vertex_id_t;

    static constexpr vertex_id_t invalid_vertex_id = BaseTraitsT::invalid_vertex_id;

    /**
     * @brief Position of this neighbor's vector in the global vector table.
     */
    vertex_id_t base_vid;

    /**
     * @brief Position of this neighbor's adjacency list in the layer-local storage.
     */
    vertex_id_t layer_vid;

    constexpr LayerNeighbor()
        : base_vid(invalid_vertex_id), layer_vid(invalid_vertex_id) {}

    constexpr LayerNeighbor(const vertex_id_t base_vid, const vertex_id_t layer_vid)
        : base_vid(base_vid), layer_vid(layer_vid) {}

    LayerNeighbor(const LayerNeighbor&) = default;
    LayerNeighbor& operator=(const LayerNeighbor&) = default;
    LayerNeighbor(LayerNeighbor&&) = default;
    LayerNeighbor& operator=(LayerNeighbor&&) = default;
    ~LayerNeighbor() = default;

    static constexpr auto make_invalid_nbr() -> LayerNeighbor {
        return LayerNeighbor{ invalid_vertex_id, invalid_vertex_id };
    }

    __attribute__((always_inline))
    constexpr bool operator==(const LayerNeighbor& other) const noexcept {
        return base_vid == other.base_vid && layer_vid == other.layer_vid;
    }

    __attribute__((always_inline))
    constexpr bool operator!=(const LayerNeighbor& other) const noexcept {
        return !(*this == other);
    }

};  // struct LayerNeighbor

}   // namespace cpu
}   // namespace artea
