// Copyright 2025 yeweitang
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
 * @FilePath: /Artea/include/artea/cpu/partitioning/vector_router.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description:
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <functional>
#include <utility>

namespace artea {
namespace cpu {

template <typename RouterTraitsT, typename DerivedClassT>
class VectorRouter {

    using vec_id_t = typename RouterTraitsT::vec_id_t;
    using vec_num_t = typename RouterTraitsT::vec_num_t;
    using vec_ele_t = typename RouterTraitsT::vec_ele_t;
    using distance_t = typename RouterTraitsT::distance_t;
    using dist_func_t = typename RouterTraitsT::dist_func_t;
    using vector_array_t = typename RouterTraitsT::vector_array_t;
    using query_vecs_t = typename RouterTraitsT::query_vecs_t;
    using idlist_array_t = typename RouterTraitsT::idlist_array_t;

public:

    VectorRouter(
        const vector_array_t& vecs_data,
        const dist_func_t& dist_func,
        const uint32_t topk
    ) :
        _num_vecs(vecs_data.get_num_vecs()),
        _vecs_data(vecs_data),
        _dist_func(dist_func),
        _topk(topk)
    {}

    /**
     * @brief Initialize the router, preparing any necessary data structures or indices (for fast routing).
     */
    template <typename... Args>
    __attribute__((always_inline))
    auto initialize(Args&&... args) -> void {
        static_cast<DerivedClassT*>(this)->initialize_impl(std::forward<Args>(args)...);
    }

    /**
     * @brief Query the top-k nearest vertices for a single vector.
     *
     * @param query_vec Pointer to the query vector data.
     * @return std::vector<vertex_id_t> Vector containing the IDs of the top-k nearest vertices.
     */
    __attribute__((always_inline))
    auto query(const vec_ele_t* query_vec) const -> std::vector<vec_id_t> {
        return static_cast<const DerivedClassT*>(this)->query_impl(query_vec);
    }

    /**
     * @brief Query the top-k nearest vertices for a single vector with an entry point.
     *
     * @param query_vec Pointer to the query vector data.
     * @param entry_point Starting vertex ID for the search.
     * @return std::vector<vertex_id_t> Vector containing the IDs of the top-k nearest vertices.
     */
    __attribute__((always_inline))
    auto query(const vec_ele_t* query_vec, const vec_id_t entry_point) const -> std::vector<vec_id_t> {
        return static_cast<const DerivedClassT*>(this)->query_impl(query_vec, entry_point);
    }

    /**
     * @brief Perform batch queries to find the top-k nearest vertices for multiple vectors.
     *
     * @param query_vecs A VectorArray containing the query vectors.
     * @return idlist_array_t Array with num_vecs=num_queries, dim=topk where each vector contains the top-k IDs for one query.
     */
    __attribute__((always_inline))
    auto batch_query(const query_vecs_t& query_vecs) const -> idlist_array_t {
        return static_cast<const DerivedClassT*>(this)->batch_query_impl(query_vecs);
    }

    /**
     * @brief Perform batch queries with a shared entry point to find the top-k nearest vertices for multiple vectors.
     *
     * @param query_vecs A VectorArray containing the query vectors.
     * @param entry_point Shared entry point vertex ID for all queries.
     * @return idlist_array_t Array with num_vecs=num_queries, dim=topk where each vector contains the top-k IDs for one query.
     */
    __attribute__((always_inline))
    auto batch_query(const query_vecs_t& query_vecs, const vec_id_t entry_point) const -> idlist_array_t {
        return static_cast<const DerivedClassT*>(this)->batch_query_impl(query_vecs, entry_point);
    }

protected:

    /** @brief Number of vectors in the dataset. */
    const vec_num_t _num_vecs;

    /** @brief Reference to the vector data. */
    const vector_array_t& _vecs_data;

    /** @brief Reference to the injected distance function functor. */
    const dist_func_t& _dist_func;

    /** @brief Number of nearest neighbors to return. */
    const uint32_t _topk;

};  // class VectorRouter

}   // namespace cpu
}   // namespace artea
