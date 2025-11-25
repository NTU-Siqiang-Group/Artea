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
 * @FilePath: /Artea/include/artea/cpu/cluster_router.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2025-11-22 17:17:43
 * @Date: 2025-11-22 10:31:36
 * @Description:
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <functional>

#include <artea/cpu/vector_array.hpp>
#include <artea/definitions.hpp>

namespace artea {
namespace cpu {

template <
    typename vertex_num_t,
    typename vec_ele_t,
    typename dist_func_t,
    typename derived_class_t
>
class ClusterRouter {

    using vertex_id_t = vertex_num_t;
    using distance_t = vec_ele_t;

public:

    ClusterRouter(
        const VectorArray<vertex_num_t, vec_ele_t>& centroids,
        const dist_func_t& dist_func
    ) :
        _num_clusters(centroids.get_num_vecs()),
        _centroids(centroids),
        _dist_func(dist_func)
    {}

    /**
     * @brief Initialize the router, preparing any necessary data structures or indices (for fast routing).
     */
    __attribute__((always_inline))
    auto initialize() -> void {
        static_cast<derived_class_t*>(this)->initialize_impl();
    }

    /**
     * @brief Query the nearest cluster for a single vector.
     *
     * @param query_vec Pointer to the query vector data.
     * @return cluster_id_t The ID of the nearest cluster.
     */
    __attribute__((always_inline))
    auto query(const vec_ele_t* query_vec) const -> cluster_id_t {
        return static_cast<const derived_class_t*>(this)->query_impl(query_vec);
    }

    /**
     * @brief Perform batch queries to find the nearest cluster for multiple vectors.
     *
     * This implementation always parallelizes the batch processing (Inter-query parallelism) using TBB
     *
     * @param query_vecs A VectorArray containing the query vectors.
     * @return std::vector<vertex_id_t> A vector containing the ID of the nearest cluster for each query.
     */
    auto batch_query(const VectorArray<vertex_num_t, vec_ele_t>& query_vecs) const -> std::vector<cluster_id_t> {
        const vertex_num_t num_queries = query_vecs.get_num_vecs();
        std::vector<vertex_id_t> results(num_queries);
        tbb::parallel_for(
            // Range: Iterate over all query vectors
            tbb::blocked_range<vertex_num_t>(0, num_queries),

            // Processor for a sub-range of queries
            [&](const tbb::blocked_range<vertex_num_t>& r) {
                for (vertex_num_t i = r.begin(); i != r.end(); ++i) {
                    // Retrieve the pointer to the current query vector
                    const vec_ele_t* current_vec = query_vecs.get(i);
                    // Call query_impl.
                    cluster_id_t best_cid = static_cast<const derived_class_t*>(this)->query_impl(current_vec);
                    // Store the result
                    results[i] = static_cast<vertex_id_t>(best_cid);
                }
            }
        );

        return results;
    }

protected:

    /** @brief Target number of clusters (K). */
    const cluster_num_t _num_clusters;

    /** @brief Reference to the centroids of the clusters. */
    const VectorArray<vertex_num_t, vec_ele_t>& _centroids;

    /** @brief Reference to the injected distance function functor. */
    const dist_func_t& _dist_func;

};  // class ClusterRouter

}   // namespace cpu
}   // namespace artea