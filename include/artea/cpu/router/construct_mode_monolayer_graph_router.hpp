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
 * @FilePath: /Artea/include/artea/cpu/router/construct_mode_monolayer_graph_router.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: construct_mode specialization of MonolayerGraphRouter.
 *               Operates on DescentGraph (dnbr_t neighbors) for build-time navigation.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include <artea/cpu/utils/parallel.hpp>
#include <artea/cpu/framework/type_traits/router_traits.hpp>
#include <artea/cpu/router/candidate_queue_concept.hpp>
#include <artea/cpu/router/visited_table_concept.hpp>

namespace artea {
namespace cpu {

template <typename RouterTraitsT>
class MonolayerGraphRouter<RouterTraitsT, GraphModeT::construct_mode> :
    public RouterTraitsT::template vector_router_t<MonolayerGraphRouter<RouterTraitsT, GraphModeT::construct_mode>>
{
    using vertex_num_t = typename RouterTraitsT::vertex_num_t;
    using vertex_id_t = typename RouterTraitsT::vertex_id_t;
    using vec_ele_t = typename RouterTraitsT::vec_ele_t;
    using distance_t = typename RouterTraitsT::distance_t;
    using dist_func_t = typename RouterTraitsT::dist_func_t;
    using vector_array_t = typename RouterTraitsT::vector_array_t;
    using query_vecs_t = typename RouterTraitsT::query_vecs_t;
    using dnbr_arr_t = typename RouterTraitsT::dnbr_arr_t;
    using candidate_queue_t = typename RouterTraitsT::candidate_queue_t;
    using visited_table_t = typename RouterTraitsT::visited_table_t;
    using visited_table_pool_t = typename RouterTraitsT::visited_table_pool_t;
    using knn_results_t = typename RouterTraitsT::knn_results_t;
    using base_class_t = typename RouterTraitsT::template vector_router_t<MonolayerGraphRouter<RouterTraitsT, GraphModeT::construct_mode>>;

public:

    MonolayerGraphRouter(
        const vector_array_t& vecs_data,
        const dist_func_t& dist_func,
        const uint32_t topk,
        const vertex_num_t candidate_queue_size = 16,
        const vertex_num_t extracted_nbr_size = 64
    ) : base_class_t(vecs_data, dist_func, topk),
        _candidate_queue_size(candidate_queue_size),
        _extracted_nbr_size(extracted_nbr_size),
        _visited_table_pool(vecs_data.get_num_vecs())
    {
        if (candidate_queue_size < topk) {
            ARTEA_ERROR(fmt::format(
                "candidate_queue_size ({}) must be >= topk ({})",
                candidate_queue_size, topk
            ));
        }
    }

    auto initialize() -> void {
        _visited_table_pool.warmup();
    }

    /**
     * @brief Query the top-k nearest vertices using the descent graph (build-time).
     * @param query_vec Pointer to the query vector data.
     * @param descent_graph The descent graph to search on.
     * @return knn_results_t Flat array of topk result entries sorted by distance.
     */
    template <typename DescentGraphT>
    __attribute__((always_inline))
    auto query(const vec_ele_t* query_vec, const DescentGraphT& descent_graph) const -> knn_results_t {
        auto& visited_table = _visited_table_pool.acquire();
        auto results = _beam_search(query_vec, visited_table, static_cast<vertex_id_t>(0), descent_graph);
        visited_table.clear();
        return results;
    }

    /**
     * @brief Query the top-k nearest vertices using the descent graph with an entry point.
     * @param query_vec Pointer to the query vector data.
     * @param entry_point Starting vertex ID for the search.
     * @param descent_graph The descent graph to search on.
     * @return knn_results_t Flat array of topk result entries sorted by distance.
     */
    template <typename DescentGraphT>
    __attribute__((always_inline))
    auto query(const vec_ele_t* query_vec, const vertex_id_t entry_point, const DescentGraphT& descent_graph) const -> knn_results_t {
        auto& visited_table = _visited_table_pool.acquire();
        auto results = _beam_search(query_vec, visited_table, entry_point, descent_graph);
        visited_table.clear();
        return results;
    }

    /**
     * @brief Perform batch queries to find the top-k nearest vertices for multiple vectors.
     * @param query_vecs A VectorArray containing the query vectors.
     * @param descent_graph The descent graph to search on.
     * @return knn_results_t Flat array of num_queries * topk result entries in row-major order.
     */
    template <typename DescentGraphT>
    auto batch_query(const query_vecs_t& query_vecs, const DescentGraphT& descent_graph) const -> knn_results_t {
        const vertex_num_t num_queries = query_vecs.get_num_vecs();
        const uint32_t K = this->_topk;
        knn_results_t results(num_queries * K);

        tbb::parallel_for(
            tbb::blocked_range<vertex_num_t>(0, num_queries),
            [&](const tbb::blocked_range<vertex_num_t>& r) {
                auto& visited = _visited_table_pool.acquire();
                for (vertex_num_t i = r.begin(); i != r.end(); ++i) {
                    const vec_ele_t* q_vec = query_vecs.get(i);
                    auto topk_results = _beam_search(q_vec, visited, static_cast<vertex_id_t>(0), descent_graph);
                    std::copy(topk_results.begin(), topk_results.end(), results.begin() + i * K);
                    visited.clear();
                }
            }
        );

        return results;
    }

private:

    /**
     * @brief Perform beam search on the descent graph with a given entry point.
     * @param query_vec Pointer to the query vector data.
     * @param visited_table Reference to the visited table for tracking explored vertices.
     * @param entry_point Starting vertex ID for the search.
     * @param descent_graph The descent graph to search on.
     * @return knn_results_t Flat array of topk result entries sorted by distance.
     */
    template <typename DescentGraphT>
    auto _beam_search(
        const vec_ele_t* query_vec,
        visited_table_t& visited_table,
        const vertex_id_t entry_point,
        const DescentGraphT& descent_graph
    ) const -> knn_results_t {
        const vertex_num_t queue_capacity = std::max(this->_topk, _candidate_queue_size);
        candidate_queue_t candidate_queue(queue_capacity);

        const distance_t entry_dist = this->_dist_func(query_vec, this->_vecs_data.get(entry_point));
        candidate_queue.try_push(entry_point, entry_dist);
        visited_table.set(entry_point);

        while (!candidate_queue.empty()) {
            if (candidate_queue.should_terminate()) { break; }
            auto [current_id, current_dist] = candidate_queue.pop_best_unexplored();
            if (current_id == RouterTraitsT::invalid_vertex_id) { break; }

            const dnbr_arr_t& nbrs = descent_graph.fetch_nbrs(current_id);
            const vertex_num_t nbr_limit = std::min(static_cast<vertex_num_t>(nbrs.size()), _extracted_nbr_size);
            for (vertex_num_t i = 0; i < nbr_limit; ++i) {
                const vertex_id_t nbr_id = nbrs[i].get_id();
                if (nbr_id == RouterTraitsT::invalid_vertex_id) { break; }
                if (visited_table.test(nbr_id)) { continue; }
                visited_table.set(nbr_id);
                const distance_t nbr_dist = this->_dist_func(query_vec, this->_vecs_data.get(nbr_id));
                candidate_queue.try_push(nbr_id, nbr_dist);
            }
        }

        return candidate_queue.extract_results(this->_topk);
    }

    /** @brief Candidate queue size for beam search. */
    vertex_num_t _candidate_queue_size = 0;

    /** @brief Maximum number of neighbors to explore per vertex. */
    vertex_num_t _extracted_nbr_size = 64;

    /** @brief Pool of thread-local visited bitmaps for parallel beam search. */
    mutable visited_table_pool_t _visited_table_pool;

};  // class MonolayerGraphRouter<RouterTraitsT, GraphModeT::construct_mode>

}   // namespace cpu
}   // namespace artea
