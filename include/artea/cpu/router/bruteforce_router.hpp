// Copyright 2025 Weitang Ye
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
 * @FilePath: /Artea/include/artea/cpu/partitioning/bruteforce_router.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2025-11-23 15:29:20
 * @Date: 2025-11-23 15:11:26
 * @Description:
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <utility>
#include <numeric>
#include <limits>

#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>

#include <artea/cpu/containers/vector_array.hpp>
#include <artea/cpu/partitioning/cluster_router.hpp>
#include <artea/common/definitions.hpp>

namespace artea {
namespace cpu {

template <typename RouterTraitsT>
class BruteforceRouter : public RouterTraitsT::cluster_router_t
{
    using vertex_num_t = typename RouterTraitsT::vertex_num_t;
    using vertex_id_t = typename RouterTraitsT::vertex_id_t;
    using vec_ele_t = typename RouterTraitsT::vec_ele_t;
    using distance_t = typename RouterTraitsT::distance_t;
    using dist_func_t = typename RouterTraitsT::dist_func_t;
    using vector_array_t = typename RouterTraitsT::vector_array_t;
    using cluster_id_t = typename RouterTraitsT::cluster_id_t;
    static constexpr bool intra_query_parallel = RouterTraitsT::intra_query_parallel;
    using base_class_t = typename RouterTraitsT::cluster_router_t;

public:

    BruteforceRouter(
        const vector_array_t& centroids,
        const dist_func_t& dist_func
    ) : base_class_t(centroids, dist_func)
    {}

    auto initialize_impl() -> void {
        // Do nothing
    }


    /**
     * @brief Query the nearest cluster centroid for a given vector.
     *
     * Depending on the template parameter `intra_query_parallel`, this function runs
     * either sequentially or in parallel using TBB to find the centroid with
     * the minimum distance.
     *
     * @param query_vec Pointer to the query vector data.
     * @return cluster_id_t The ID of the nearest cluster.
     */
    auto query_impl(const vec_ele_t* query_vec) const -> cluster_id_t {
        if constexpr (not intra_query_parallel) {
            // Find the cluster with the minimum distance to the query vector
            distance_t min_dist = std::numeric_limits<distance_t>::max();
            cluster_id_t best_cluster = 0;
            for (cluster_id_t cid = 0; cid < this->_num_clusters; ++cid) {
                const vec_ele_t* center = this->_centroids.get(cid);
                distance_t dist = this->_dist_func(query_vec, center);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = cid;
                }
            }
            return best_cluster;

        } else {
            // Parallel reduction to find the cluster with the minimum distance

            distance_t global_min_dist = std::numeric_limits<distance_t>::max();
            cluster_id_t global_best_cluster = 0;
            // Define a helper struct to hold the reduction result (distance + index)
            struct Result {
                distance_t min_dist;
                cluster_id_t cluster_id;
            };

            // Execute parallel reduction
            Result final_res = tbb::parallel_reduce(
                // Range: Iterate over all clusters
                tbb::blocked_range<cluster_id_t>(0, this->_num_clusters),

                // Identity value: Max distance
                Result { std::numeric_limits<distance_t>::max(), 0 },

                // Processor for a sub-range of clusters
                [&](const tbb::blocked_range<cluster_id_t>& r, Result local_res) -> Result {
                    for (cluster_id_t cid = r.begin(); cid != r.end(); ++cid) {
                        const vec_ele_t* center = this->_centroids.get(cid);
                        distance_t dist = this->_dist_func(query_vec, center);

                        if (dist < local_res.min_dist) {
                            local_res.min_dist = dist;
                            local_res.cluster_id = cid;
                        }
                    }
                    return local_res;
                },

                // Join Operator: Merge results from two threads (keep the one with smaller distance)
                [](const Result& a, const Result& b) -> Result {
                    return (a.min_dist < b.min_dist) ? a : b;
                }
            );

            global_best_cluster = final_res.cluster_id;

            return global_best_cluster;
        }
    }

};  // class BruteforceRouter

}   // namespace cpu
}   // namespace artea