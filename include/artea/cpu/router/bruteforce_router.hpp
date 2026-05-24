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
    using base_norms_t = typename RouterTraitsT::base_norms_t;
    using base_class_t = typename RouterTraitsT::template vector_router_t<BruteforceRouter<RouterTraitsT>>;

    static constexpr bool intra_query_parallel = RouterTraitsT::intra_query_parallel;

public:

    BruteforceRouter(
        const vector_array_t& vecs_data,
        const dist_func_t& dist_func,
        const uint32_t topk
    ) : base_class_t(vecs_data, dist_func, topk) {}

    auto initialize() -> void {
        // Do nothing
    }


    /**
     * @brief Query the top-k nearest vertices via plain L2 (or whatever
     *        symmetric metric the dist_func is instantiated with).
     */
    auto query(const vec_ele_t* query_vec) const -> knn_results_t {
        const uint32_t k = std::min(this->_topk, static_cast<uint32_t>(this->_num_vecs));

        knn_results_t all_results;
        all_results.reserve(this->_num_vecs);
        for (vertex_id_t vid = 0; vid < this->_num_vecs; ++vid) {
            const distance_t dist = this->_dist_func(query_vec, this->_vecs_data.get(vid));
            all_results.emplace_back(vid, dist);
        }

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
     * @brief Bruteforce variant on the FastL2 path. Computes
     *        -2*<p, q> + ||p||^2 per candidate via @c dist_func.fast_euclidean.
     *        Result is ranking-equivalent to @c query() within a single
     *        query (drops the per-query constant ||q||^2) — top-K vid sets
     *        match @c query()'s up to FP tie-break.
     *
     * @param query_vec  Query vector.
     * @param base_norms Per-base ||p||^2 cache, indexed by vid
     *                   (from VectorDataset::get_base_norms() after
     *                    enable_fast_L2()).
     */
    auto query_fast(const vec_ele_t* query_vec,
                    const base_norms_t& base_norms) const -> knn_results_t
    {
        const uint32_t k = std::min(this->_topk, static_cast<uint32_t>(this->_num_vecs));

        knn_results_t all_results;
        all_results.reserve(this->_num_vecs);
        for (vertex_id_t vid = 0; vid < this->_num_vecs; ++vid) {
            const distance_t dist = this->_dist_func.fast_euclidean(
                this->_vecs_data.get(vid),
                query_vec,
                base_norms[vid]
            );
            all_results.emplace_back(vid, dist);
        }

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
     * @brief Entry-point overload of @c query — entry_point is ignored
     *        (bruteforce visits every vertex anyway). Kept for API
     *        parity with other routers.
     */
    auto query(const vec_ele_t* query_vec, const vertex_id_t /*entry_point*/) const -> knn_results_t {
        return query(query_vec);
    }

    /**
     * @brief Perform batch queries (L2 path).
     *
     * Always parallelizes over queries with TBB. Results laid out flat
     * as num_queries * topk in row-major order.
     */
    auto batch_query(const typename RouterTraitsT::query_vecs_t& query_vecs) const -> knn_results_t {
        const vertex_num_t num_queries = query_vecs.get_num_vecs();
        const uint32_t K = this->_topk;

        knn_results_t results(num_queries * K);

        tbb::parallel_for(
            tbb::blocked_range<vertex_num_t>(0, num_queries),
            [&](const tbb::blocked_range<vertex_num_t>& r) {
                for (vertex_num_t i = r.begin(); i != r.end(); ++i) {
                    const vec_ele_t* current_vec = query_vecs.get(i);
                    auto topk_results = this->query(current_vec);
                    std::copy(topk_results.begin(), topk_results.end(), results.begin() + i * K);
                }
            }
        );

        return results;
    }

    /**
     * @brief Perform batch queries on the FastL2 path. See @c query_fast.
     */
    auto batch_query_fast(const typename RouterTraitsT::query_vecs_t& query_vecs,
                          const base_norms_t& base_norms) const -> knn_results_t
    {
        const vertex_num_t num_queries = query_vecs.get_num_vecs();
        const uint32_t K = this->_topk;

        knn_results_t results(num_queries * K);

        tbb::parallel_for(
            tbb::blocked_range<vertex_num_t>(0, num_queries),
            [&](const tbb::blocked_range<vertex_num_t>& r) {
                for (vertex_num_t i = r.begin(); i != r.end(); ++i) {
                    const vec_ele_t* current_vec = query_vecs.get(i);
                    auto topk_results = this->query_fast(current_vec, base_norms);
                    std::copy(topk_results.begin(), topk_results.end(), results.begin() + i * K);
                }
            }
        );

        return results;
    }

    /**
     * @brief Entry-point overload of @c batch_query — entry_point is
     *        ignored. Kept for API parity.
     */
    auto batch_query(const typename RouterTraitsT::query_vecs_t& query_vecs,
                     const vertex_id_t /*entry_point*/) const -> knn_results_t {
        return batch_query(query_vecs);
    }

};  // class BruteforceRouter

}   // namespace cpu
}   // namespace artea
