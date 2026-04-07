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
 * @FilePath: /Artea/include/artea/cpu/index/hierarchy_manager.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Date: 2026-03-10
 * @Description: Hierarchical vector manager for HNSW-like index.
 */

#pragma once

#include <vector>
#include <utility>
#include <artea/common/logger.hpp>

namespace artea {
namespace cpu {

/**
 * @brief Hierarchical vector manager for HNSW-like index.
 * @tparam IndexTraitsT The index traits type.
 *
 * @note Layer ID mapping:
 *   - layer_id 0 is the bottom layer (base_vecs, const reference)
 *   - layer_id > 0 are upper layers stored in _upper_layer_vecs[layer_id - 1]
 */
template <typename IndexTraitsT>
class HierarchyManager {

    using vertex_num_t = typename IndexTraitsT::vertex_num_t;
    using layer_id_t = typename IndexTraitsT::layer_id_t;
    using layer_num_t = typename IndexTraitsT::layer_num_t;
    using vec_dim_t = typename IndexTraitsT::vec_dim_t;
    using vector_array_t = typename IndexTraitsT::vector_array_t;
    using vertex_subset_t = typename IndexTraitsT::vertex_subset_t;

public:
    /**
     * @brief Construct a new Hierarchical Vecs Manager object.
     * @param bottom_layer_vecs Reference to the base layer vector data.
     */
    explicit HierarchyManager(const vector_array_t& bottom_layer_vecs)
        : _bottom_layer_vecs(bottom_layer_vecs) {}

    // Copying is deleted
    HierarchyManager(const HierarchyManager&) = delete;
    HierarchyManager& operator=(const HierarchyManager&) = delete;

    // Default move constructor and assignment
    HierarchyManager(HierarchyManager&&) noexcept = default;
    HierarchyManager& operator=(HierarchyManager&&) noexcept = default;

    // --- Public Interface ---

    /**
     * @brief Get the base layer vector data.
     * @return const vector_array_t& Reference to the base layer vectors.
     */
    __attribute__((always_inline))
    auto get_base_vecs() const -> const vector_array_t& {
        return _bottom_layer_vecs;
    }

    /**
     * @brief Get the number of base vectors.
     * @return vertex_num_t Number of base vectors.
     */
    __attribute__((always_inline))
    auto get_num_base_vecs() const -> vertex_num_t {
        return static_cast<vertex_num_t>(_bottom_layer_vecs.get_num_vecs());
    }

    /**
     * @brief Get the number of layers (including base layer).
     * @return layer_num_t Number of layers.
     */
    __attribute__((always_inline))
    auto get_num_layers() const -> layer_num_t {
        return static_cast<layer_num_t>(_upper_layer_vecs.size()) + 1;
    }

    /**
     * @brief Get the upper layer vector arrays.
     * @return std::vector<vector_array_t>& Reference to the upper layer vectors.
     */
    __attribute__((always_inline))
    auto get_upper_layer_vecs() -> std::vector<vector_array_t>& {
        return _upper_layer_vecs;
    }

    /**
     * @brief Get the upper layer vector arrays (const version).
     * @return const std::vector<vector_array_t>& Const reference to the upper layer vectors.
     */
    __attribute__((always_inline))
    auto get_upper_layer_vecs() const -> const std::vector<vector_array_t>& {
        return _upper_layer_vecs;
    }

    /**
     * @brief Get vector data for a specific layer.
     * @param layer_id The layer ID (0 for base layer, 1+ for upper layers).
     * @return const vector_array_t& Reference to the layer's vector data.
     */
    __attribute__((always_inline))
    auto get_layer_vecs(const layer_id_t layer_id) const -> const vector_array_t& {
        if (layer_id == 0) {
            return _bottom_layer_vecs;
        }
        return _upper_layer_vecs[layer_id - 1];
    }

    /**
     * @brief Set vector data for a specific upper layer (layer_id must be > 0).
     * @param layer_id The layer ID (must be > 0, as layer 0 is the const base_vecs).
     * @param vertex_subset The vertex subset to set (move semantics).
     */
    auto set_layer_vecs(const layer_id_t layer_id, vertex_subset_t&& vertex_subset) -> void {
        #ifndef NDEBUG
        if (layer_id < 1) {
            ARTEA_ERROR("layer_id must be >= 1 for set_layer_vecs");
        }
        #endif
        _upper_layer_vecs[layer_id - 1] = std::move(vertex_subset.vecs_data);
    }

    /**
     * @brief Set vector data for a specific upper layer (layer_id must be > 0).
     * @param layer_id The layer ID (must be > 0, as layer 0 is the const base_vecs).
     * @param vecs_data The vector array to set (move semantics).
     */
    auto set_layer_vecs(const layer_id_t layer_id, vector_array_t&& vecs_data) -> void {
        #ifndef NDEBUG
        if (layer_id < 1) {
            ARTEA_ERROR("layer_id must be >= 1 for set_layer_vecs");
        }
        #endif
        _upper_layer_vecs[layer_id - 1] = std::move(vecs_data);
    }

    /**
     * @brief Append a new upper layer from bottom to top.
     * @param vertex_subset The vertex subset to append (move semantics).
     */
    auto bottom_up_append(vertex_subset_t&& vertex_subset) -> void {
        _upper_layer_vecs.push_back(std::move(vertex_subset.vecs_data));
    }

    /**
     * @brief Append a new upper layer from bottom to top.
     * @param vecs_data The vector array to append (move semantics).
     */
    auto bottom_up_append(vector_array_t&& vecs_data) -> void {
        _upper_layer_vecs.push_back(std::move(vecs_data));
    }

protected:
    /** @brief Const reference to base layer vector data. */
    const vector_array_t& _bottom_layer_vecs;

    /** @brief Upper layer vector arrays. */
    std::vector<vector_array_t> _upper_layer_vecs;

};  // class HierarchyManager

}   // namespace cpu
}   // namespace artea
