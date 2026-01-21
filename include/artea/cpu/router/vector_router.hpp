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

#include <artea/definitions.hpp>

namespace artea {
namespace cpu {

template <typename ComputerTraitsT, typename DerivedClassT>
class VectorRouter {

    using vec_id_t = typename ComputerTraitsT::vec_id_t;
    using vec_num_t = typename ComputerTraitsT::vec_num_t;
    using vec_ele_t = typename ComputerTraitsT::vec_ele_t;
    using distance_t = typename ComputerTraitsT::distance_t;
    using dist_func_t = typename ComputerTraitsT::dist_func_t;
    using vector_array_t = typename ComputerTraitsT::vector_array_t;
    using base_vecs_t = typename ComputerTraitsT::base_vecs_t;

public:

    VectorRouter(
        const base_vecs_t& base_vecs,
        const dist_func_t& dist_func
    ) :
        _num_vecs(base_vecs.get_num_vecs()),
        _base_vecs(base_vecs),
        _dist_func(dist_func)
    {}

    /**
     * @brief Initialize the router, preparing any necessary data structures or indices (for fast routing).
     */
    __attribute__((always_inline))
    auto initialize() -> void {
        static_cast<DerivedClassT*>(this)->initialize_impl();
    }

    /**
     * @brief Query the nearest vertex for a single vector.
     *
     * @param query_vec Pointer to the query vector data.
     * @return vec_id_t The ID of the nearest vertex.
     */
    __attribute__((always_inline))
    auto query(const vec_ele_t* query_vec) const -> vec_id_t {
        return static_cast<const DerivedClassT*>(this)->query_impl(query_vec);
    }

    /**
     * @brief Perform batch queries to find the nearest vertex for multiple vectors.
     *
     * This implementation always parallelizes the batch processing (Inter-query parallelism) using TBB
     *
     * @param query_vecs A VectorArray containing the query vectors.
     * @return std::vector<vec_id_t> A vector containing the ID of the nearest vertex for each query.
     */
    auto batch_query(const VectorArray<vec_num_t, vec_ele_t>& query_vecs) const -> std::vector<vec_id_t> {
        const vec_num_t num_queries = query_vecs.get_num_vecs();
        std::vector<vec_id_t> results(num_queries);
        tbb::parallel_for(
            // Range: Iterate over all query vectors
            tbb::blocked_range<vec_num_t>(0, num_queries),

            // Processor for a sub-range of queries
            [&](const tbb::blocked_range<vec_num_t>& r) {
                for (vec_num_t i = r.begin(); i != r.end(); ++i) {
                    // Retrieve the pointer to the current query vector
                    const vec_ele_t* current_vec = query_vecs.get(i);
                    // Call query_impl.
                    vec_id_t best_vid = static_cast<const DerivedClassT*>(this)->query_impl(current_vec);
                    // Store the result
                    results[i] = static_cast<vec_id_t>(best_vid);
                }
            }
        );

        return results;
    }

protected:

    /** @brief Target number of vertices (K). */
    const vec_num_t _num_vecs;

    /** @brief Reference to the target base vectors. */
    const base_vecs_t& _base_vecs;

    /** @brief Reference to the injected distance function functor. */
    const dist_func_t& _dist_func;

};  // class VectorRouter

}   // namespace cpu
}   // namespace artea