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
#include <artea/cpu/partitioning/vector_router.hpp>
#include <artea/common/definitions.hpp>

namespace artea {
namespace cpu {

template <typename ComputerTraitsT, bool IntraQueryParallel = false>
class BruteforceRouter :
    public VectorRouter<ComputerTraitsT, BruteforceRouter<ComputerTraitsT, IntraQueryParallel>>
{
    using vec_num_t = typename ComputerTraitsT::vec_num_t;
    using vec_id_t = typename ComputerTraitsT::vec_id_t;
    using vec_ele_t = typename ComputerTraitsT::vec_ele_t;
    using distance_t = typename ComputerTraitsT::distance_t;
    using dist_func_t = typename ComputerTraitsT::dist_func_t;
    using vector_array_t = typename ComputerTraitsT::vector_array_t;
    using base_vecs_t = typename ComputerTraitsT::base_vecs_t;
    using base_class_t = VectorRouter<ComputerTraitsT, BruteforceRouter<ComputerTraitsT, IntraQueryParallel>>;

public:

    BruteforceRouter(
        const base_vecs_t& base_vecs,
        const dist_func_t& dist_func
    ) : base_class_t(base_vecs, dist_func)
    {}

    auto initialize_impl() -> void {
        // Do nothing
    }


    /**
     * @brief Query the nearest vertex centroid for a given vector.
     *
     * Depending on the template parameter `IntraQueryParallel`, this function runs
     * either sequentially or in parallel using TBB to find the centroid with
     * the minimum distance.
     *
     * @param query_vec Pointer to the query vector data.
     * @return vec_id_t The ID of the nearest vertex.
     */
    auto query_impl(const vec_ele_t* query_vec) const -> vec_id_t {
        if constexpr (not IntraQueryParallel) {
            // Find the vertex with the minimum distance to the query vector
            distance_t min_dist = std::numeric_limits<distance_t>::max();
            vec_id_t best_vid = 0;
            for (vec_id_t vid = 0; vid < this->_num_vecs; ++vid) {
                const vec_ele_t* vec = this->_base_vecs.get(vid);
                distance_t dist = this->_dist_func(query_vec, vec);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_vid = vid;
                }
            }
            return best_vid;
        }
        else {
            // Parallel reduction to find the vertex with the minimum distance

            distance_t global_min_dist = std::numeric_limits<distance_t>::max();
            vec_id_t global_best_vid = 0;
            // Define a helper struct to hold the reduction result (distance + index)
            struct Result {
                distance_t min_dist;
                vec_id_t vertex_id;
            };

            // Execute parallel reduction
            Result final_res = tbb::parallel_reduce(
                // Range: Iterate over all vertices
                tbb::blocked_range<vec_id_t>(0, this->_num_vecs),

                // Identity value: Max distance
                Result { std::numeric_limits<distance_t>::max(), 0 },

                // Processor for a sub-range of vertices
                [&](const tbb::blocked_range<vec_id_t>& r, Result local_res) -> Result {
                    for (vec_id_t vid = r.begin(); vid != r.end(); ++vid) {
                        const vec_ele_t* vec = this->_base_vecs.get(vid);
                        distance_t dist = this->_dist_func(query_vec, vec);

                        if (dist < local_res.min_dist) {
                            local_res.min_dist = dist;
                            local_res.vertex_id = vid;
                        }
                    }
                    return local_res;
                },

                // Join Operator: Merge results from two threads (keep the one with smaller distance)
                [](const Result& a, const Result& b) -> Result {
                    return (a.min_dist < b.min_dist) ? a : b;
                }
            );

            global_best_vid = final_res.vertex_id;

            return global_best_vid;
        }
    }

};  // class BruteforceRouter

}   // namespace cpu
}   // namespace artea