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
 * @FilePath: /Artea/include/artea/cpu/containers/vertex_subset.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: VertexSubset container for vertex generation algorithms.
 */

#pragma once

#include <artea/cpu/containers/allocator.hpp>

namespace artea {
namespace cpu {

/**
 * @brief VertexSubset represents a subset of vertices from vertex generation algorithms.
 * @tparam BaseTraitsT The base traits type
 */
template <typename BaseTraitsT>
struct VertexSubset {
    using vec_id_t = typename BaseTraitsT::vec_id_t;
    using vec_num_t = typename BaseTraitsT::vec_num_t;
    using vec_dim_t = typename BaseTraitsT::vec_dim_t;
    using vector_array_t = typename BaseTraitsT::vector_array_t;

    /** @brief Vector IDs in the vertex subset (cache-aligned for performance) */
    cache_aligned_container_t<vec_id_t> vec_ids;

    /** @brief Vector data corresponding to the IDs */
    vector_array_t vecs_data;

    /** @brief Default constructor */
    VertexSubset() = default;

    /** @brief Constructor with dimension */
    explicit VertexSubset(vec_dim_t dim)
        : vecs_data(dim) {}

    // Copying is deleted
    VertexSubset(const VertexSubset&) = delete;
    VertexSubset& operator=(const VertexSubset&) = delete;

    // default move constructor and assignment
    VertexSubset(VertexSubset&&) noexcept = default;
    VertexSubset& operator=(VertexSubset&&) noexcept = default;

    /** @brief Get the number of vectors in the subset */
    auto get_num_vecs() const -> vec_num_t {
        return vecs_data.get_num_vecs();
    }

    /** @brief Get the number of vectors in the subset (alias for get_num_vecs) */
    auto size() const -> vec_num_t {
        return get_num_vecs();
    }

    /** @brief Reserve space for n vectors */
    void reserve(vec_num_t n) {
        vec_ids.reserve(n);
        vecs_data.reserve(n);
    }
};

}   // namespace cpu
}   // namespace artea