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
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>

namespace artea {
namespace cpu {

template <typename RouterTraitsT>
class BruteforceRouter :
    public RouterTraitsT::template vector_router_t<BruteforceRouter<RouterTraitsT>>
{
    using vertex_num_t = typename RouterTraitsT::vertex_num_t;
    using vertex_id_t = typename RouterTraitsT::vertex_id_t;
    using vec_ele_t = typename RouterTraitsT::vec_ele_t;
    using distance_t = typename RouterTraitsT::distance_t;
    using dist_func_t = typename RouterTraitsT::dist_func_t;
    using vector_array_t = typename RouterTraitsT::vector_array_t;
    using result_entry_t = typename RouterTraitsT::result_entry_t;
    using knn_results_t = typename RouterTraitsT::knn_results_t;
    using base_class_t = typename RouterTraitsT::template vector_router_t<BruteforceRouter<RouterTraitsT>>;

    static constexpr bool intra_query_parallel = RouterTraitsT::intra_query_parallel;

public:

    BruteforceRouter(
        const vector_array_t& vecs_data,
        const dist_func_t& dist_func,
        const uint32_t topk
    ) : base_class_t(vecs_data, dist_func, topk)
    {}

    auto initialize_impl() -> void {
        // Do nothing
    }


    /**
     * @brief Query the top-k nearest vertices for a given vector.
     *
     * @param query_vec Pointer to the query vector data.
     * @return std::vector<vec_id_t> Vector containing the IDs of the top-k nearest vertices.
     */
    auto query_impl(const vec_ele_t* query_vec) const -> knn_results_t {
        const uint32_t k = std::min(this->_topk, static_cast<uint32_t>(this->_num_vecs));

        // Compute distances for all vertices
        knn_results_t all_results;
        all_results.reserve(this->_num_vecs);
        for (vertex_id_t vid = 0; vid < this->_num_vecs; ++vid) {
            distance_t dist = this->_dist_func(query_vec, this->_vecs_data.get(vid));
            all_results.emplace_back(vid, dist);
        }

        // Partial sort to get top-k smallest distances
        std::partial_sort(
            all_results.begin(),
            all_results.begin() + k,
            all_results.end(),
            [](const result_entry_t& a, const result_entry_t& b) {
                return a.get_distance() < b.get_distance();
            }
        );

        all_results.resize(k);
        return all_results;
    }

    /**
     * @brief Query the top-k nearest vertices for a given vector with an entry point.
     *
     * For bruteforce router, entry point is ignored and this delegates to the standard query_impl.
     *
     * @param query_vec Pointer to the query vector data.
     * @param entry_point Starting vertex ID (ignored for bruteforce).
     * @return std::vector<vec_id_t> Vector containing the IDs of the top-k nearest vertices.
     */
    auto query_impl(const vec_ele_t* query_vec, const vertex_id_t entry_point) const -> knn_results_t {
        // For bruteforce, entry point doesn't matter - just delegate to standard implementation
        return query_impl(query_vec);
    }

    /**
     * @brief Perform batch queries to find the top-k nearest vertices for multiple vectors.
     *
     * This implementation always parallelizes the batch processing (Inter-query parallelism) using TBB.
     * Results are stored as vectors: each query's k nearest neighbors form a single vector.
     *
     * @param query_vecs A VectorArray containing the query vectors.
     * @return knn_results_t Flat array of num_queries * topk result entries in row-major order.
     */
    auto batch_query_impl(const typename RouterTraitsT::query_vecs_t& query_vecs) const -> knn_results_t {
        const vertex_num_t num_queries = query_vecs.get_num_vecs();
        const uint32_t K = this->_topk;

        // Pre-allocate flat result array (num_queries * K entries)
        knn_results_t results(num_queries * K);

        tbb::parallel_for(
            // Range: Iterate over all query vectors
            tbb::blocked_range<vertex_num_t>(0, num_queries),

            // Processor for a sub-range of queries
            [&](const tbb::blocked_range<vertex_num_t>& r) {
                for (vertex_num_t i = r.begin(); i != r.end(); ++i) {
                    // Retrieve the pointer to the current query vector
                    const vec_ele_t* current_vec = query_vecs.get(i);
                    // Call query_impl to get top-k results
                    auto topk_results = this->query_impl(current_vec);
                    // Store results into flat array at row i
                    std::copy(topk_results.begin(), topk_results.end(), results.begin() + i * K);
                }
            }
        );

        return results;
    }

    /**
     * @brief Perform batch queries with a shared entry point to find the top-k nearest vertices for multiple vectors.
     *
     * For bruteforce router, entry point is ignored and this delegates to the standard batch_query_impl.
     *
     * @param query_vecs A VectorArray containing the query vectors.
     * @param entry_point Shared entry point vertex ID (ignored for bruteforce).
     * @return knn_results_t Flat array of num_queries * topk result entries in row-major order.
     */
    auto batch_query_impl(const typename RouterTraitsT::query_vecs_t& query_vecs, const vertex_id_t entry_point) const -> knn_results_t {
        // For bruteforce, entry point doesn't matter - just delegate to standard implementation
        return batch_query_impl(query_vecs);
    }

};  // class BruteforceRouter

}   // namespace cpu
}   // namespace artea