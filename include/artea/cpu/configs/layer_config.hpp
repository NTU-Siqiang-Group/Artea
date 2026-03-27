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
 * @FilePath: /Artea/include/artea/cpu/configs/layer_config.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Configuration for graph layer parameters.
 */

#pragma once

namespace artea {
namespace cpu {

/**
 * @brief Configuration for a single layer in hierarchical graph.
 * Supports builder pattern for flexible configuration.
 * @tparam BaseTraitsT The index traits type.
 */
template <typename BaseTraitsT>
struct LayerConfig {
    using vertex_num_t = typename BaseTraitsT::vertex_num_t;

    /**
     * @brief Constructor for layer configuration.
     * @param max_nbr_size Maximum number of neighbors (for overflow control).
     * @param reserved_nbr_size Reserved neighbor array size.
     */
    LayerConfig(
        vertex_num_t max_nbr_size,
        vertex_num_t reserved_nbr_size
    ) :
        _max_nbr_size(max_nbr_size),
        _reserved_nbr_size(reserved_nbr_size)
    {}

    // Builder pattern setters (chainable)
    auto max_nbr_size(vertex_num_t value) -> LayerConfig& { _max_nbr_size = value; return *this; }
    auto reserved_nbr_size(vertex_num_t value) -> LayerConfig& { _reserved_nbr_size = value; return *this; }

    // Const getters
    auto max_nbr_size() const -> vertex_num_t { return _max_nbr_size; }
    auto reserved_nbr_size() const -> vertex_num_t { return _reserved_nbr_size; }

private:
    /** @brief Maximum number of neighbors (for overflow control). */
    vertex_num_t _max_nbr_size;

    /** @brief Reserved neighbor array size. */
    vertex_num_t _reserved_nbr_size;
};

}   // namespace cpu
}   // namespace artea
