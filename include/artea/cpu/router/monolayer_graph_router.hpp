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
 * @FilePath: /Artea/include/artea/cpu/router/monolayer_graph_router.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description:
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <functional>
#include <algorithm>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include <artea/cpu/utils/parallel.hpp>
#include <artea/cpu/router/candidate_queue_concept.hpp>
#include <artea/cpu/router/visited_table_concept.hpp>

namespace artea {
namespace cpu {

template <
    typename RouterTraitsT,
    CandidateQueue CandidateQueueImpl = typename RouterTraitsT::std_candidate_queue_t,
    VisitedTable VisitedTableImpl = typename RouterTraitsT::thread_local_bitmap_t
>   requires CandidateQueue<CandidateQueueImpl> && VisitedTable<VisitedTableImpl>
class MonolayerGraphRouter :
    public RouterTraitsT::template vector_router_t<MonolayerGraphRouter<RouterTraitsT, CandidateQueueImpl, VisitedTableImpl>>
{

    using candidate_queue_t = CandidateQueueImpl;
    using visited_table_t = VisitedTableImpl;
    using visited_table_pool_t = typename RouterTraitsT::template visited_table_pool_t<VisitedTableImpl>;
    using vertex_num_t = typename RouterTraitsT::vertex_num_t;
    using vertex_id_t = typename RouterTraitsT::vertex_id_t;
    using vec_ele_t = typename RouterTraitsT::vec_ele_t;
    using distance_t = typename RouterTraitsT::distance_t;
    using dist_func_t = typename RouterTraitsT::dist_func_t;
    using vector_array_t = typename RouterTraitsT::vector_array_t;
    using idlist_array_t = typename RouterTraitsT::idlist_array_t;
    using flat_search_graph_t = typename RouterTraitsT::flat_search_graph_t;
    using random_seq_t = typename RouterTraitsT::random_seq_t;
    using base_class_t = typename RouterTraitsT::template vector_router_t<MonolayerGraphRouter<RouterTraitsT, CandidateQueueImpl, VisitedTableImpl>>;
    static constexpr bool intra_query_parallel = RouterTraitsT::intra_query_parallel;

public:

    MonolayerGraphRouter(
        const vector_array_t& vecs_data,
        const dist_func_t& dist_func,
        const flat_search_graph_t& flat_search_graph,
        const uint32_t topk,
        const vertex_num_t candidate_queue_size = 16
    ) : base_class_t(vecs_data, dist_func, topk),
        _flat_search_graph(flat_search_graph),
        _candidate_queue_size(candidate_queue_size),
        _visited_table_pool(vecs_data.get_num_vecs()),
        _random_seq(vecs_data.get_num_vecs())
    {}

    auto initialize_impl() -> void {
        _visited_table_pool.warmup();
        _warmup_random_seq();
    }

    /**
     * @brief Query the top-k nearest vertices using proximity graph.
     * @param query_vec Pointer to the query vector data.
     * @return std::vector<vertex_id_t> Vector containing the IDs of the top-k nearest vertices.
     */
    __attribute__((always_inline))
    auto query_impl(const vec_ele_t* query_vec) const -> std::vector<vertex_id_t> {
        auto& visited_table = _visited_table_pool.acquire();
        std::vector<vertex_id_t> result_ids = beam_search(
            query_vec,
            visited_table,
            _random_seq
        );
        return result_ids;
    }

    /**
     * @brief Perform batch queries to find the top-k nearest vertices for multiple vectors.
     *
     * This implementation always parallelizes the batch processing (Inter-query parallelism) using TBB.
     * Results are stored as vectors: each query's k nearest neighbors form a single vector.
     *
     * @param query_vecs A VectorArray containing the query vectors.
     * @return idlist_array_t Array with num_vecs=num_queries, dim=topk where each vector contains the top-k IDs for one query.
     */
    auto batch_query_impl(const typename RouterTraitsT::query_vecs_t& query_vecs) const -> idlist_array_t {
        const vertex_num_t num_queries = query_vecs.get_num_vecs();

        // Pre-allocate the result container (num_queries vectors, each with dimension = topk)
        idlist_array_t results(num_queries, this->_topk);

        tbb::parallel_for(
            // Range: Iterate over all query vectors
            tbb::blocked_range<vertex_num_t>(0, num_queries),

            // Processor for a sub-range of queries
            [&](const tbb::blocked_range<vertex_num_t>& r) {
                auto& visited = _visited_table_pool.acquire();
                for (vertex_num_t i = r.begin(); i != r.end(); ++i) {
                    const vec_ele_t* q_vec = query_vecs.get(i);
                    // Call query_impl to get top-k results (returns std::vector<vertex_id_t>)
                    auto topk_results = this->query_impl(q_vec);
                    // Store results using VectorArray's set interface
                    results.set(i, topk_results.data());
                    visited.clear();
                }
            }
        );

        return results;
    }

    /**
     * @brief Perform beam search on the proximity graph to find top-k nearest neighbors.
     * @param query_vec Pointer to the query vector data.
     * @param visited_table Reference to the visited table for tracking explored vertices.
     * @param entry_point Starting vertex ID for the search.
     * @return Vector of vertex IDs representing the top-k nearest neighbors.
     */
    auto beam_search(
        const vec_ele_t* query_vec,
        visited_table_t& visited_table,
        const vertex_id_t entry_point
    ) const -> std::vector<vertex_id_t> {
        using candidate_entry_t = typename RouterTraitsT::candidate_entry_t;

        // Initialize candidate queue with capacity = max(topk, _candidate_queue_size)
        const vertex_num_t queue_capacity = std::max(this->_topk, _candidate_queue_size);
        candidate_queue_t candidate_queue(queue_capacity);

        // Compute distance to entry point and initialize queue
        const distance_t entry_dist = this->_dist_func(query_vec, this->_vecs_data.get(entry_point));

        candidate_queue.try_push(entry_point, entry_dist);
        visited_table.set(entry_point);

        // Beam search loop
        while (!candidate_queue.empty()) {
            // Termination check: if current vertex distance > lower bound and queue is full
            if (candidate_queue.should_terminate()) { break; }
            // Get the best unexplored candidate
            auto [current_id, current_dist] = candidate_queue.pop_best_unexplored();
            // Check for invalid entry or early termination
            if (current_id == RouterTraitsT::invalid_vertex_id) { break; }
            // Explore neighbors of current vertex
            const vertex_id_t* neighbors = _flat_search_graph.get_neighbors(current_id);
            const vertex_num_t nbr_count = _flat_search_graph.get_extracted_nbr_size();

            for (vertex_num_t i = 0; i < nbr_count; ++i) {
                const vertex_id_t nbr_id = neighbors[i];
                // Skip invalid neighbors
                if (nbr_id == RouterTraitsT::invalid_vertex_id) { continue; }
                // Skip already visited neighbors
                if (visited_table.test(nbr_id)) { continue; }
                // Mark as visited
                visited_table.set(nbr_id);
                // Compute distance to neighbor
                const distance_t nbr_dist = this->_dist_func(query_vec, this->_vecs_data.get(nbr_id));
                // Try to add neighbor to candidate queue
                candidate_queue.try_push(nbr_id, nbr_dist);
            }
        }

        // Extract top-k result IDs from candidate queue
        return candidate_queue.extract_result_ids(this->_topk);
    }

    /**
     * @brief Perform beam search without entry point using random initialization.
     * @param query_vec Pointer to the query vector data.
     * @param visited_table Reference to the visited table for tracking explored vertices.
     * @param random_seq Random sequence generator for generating random vertex IDs (thread-safe).
     * @return Vector of vertex IDs representing the top-k nearest neighbors.
     * @note This function uses random_initialize to initialize the candidate queue with random vertices.
     */
    __attribute__((always_inline))
    auto beam_search(
        const vec_ele_t* query_vec,
        visited_table_t& visited_table,
        random_seq_t& random_seq
    ) const -> std::vector<vertex_id_t> {
        using candidate_entry_t = typename RouterTraitsT::candidate_entry_t;

        // Initialize candidate queue with capacity = max(topk, _candidate_queue_size)
        const vertex_num_t queue_capacity = std::max(this->_topk, _candidate_queue_size);
        candidate_queue_t candidate_queue(queue_capacity);

        // Initialize candidate queue with random vertices
        candidate_queue.random_initialize(
            random_seq,
            this->_dist_func,
            query_vec,
            this->_vecs_data
        );

        // Mark all initial candidates as visited
        // We need to extract the initial candidates from the queue to mark them as visited
        // Since we can't directly access the queue's internal data, we'll mark them during the search loop

        // Beam search loop
        while (!candidate_queue.empty()) {
            // Termination check: if current vertex distance > lower bound and queue is full
            if (candidate_queue.should_terminate()) { break; }
            // Get the best unexplored candidate
            auto [current_id, current_dist] = candidate_queue.pop_best_unexplored();
            // Check for invalid entry or early termination
            if (current_id == RouterTraitsT::invalid_vertex_id) { break; }

            // Mark as visited if not already visited
            if (!visited_table.test(current_id)) {
                visited_table.set(current_id);
            }

            // Explore neighbors of current vertex
            const vertex_id_t* neighbors = _flat_search_graph.get_neighbors(current_id);
            const vertex_num_t nbr_count = _flat_search_graph.get_extracted_nbr_size();

            for (vertex_num_t i = 0; i < nbr_count; ++i) {
                const vertex_id_t nbr_id = neighbors[i];
                // Skip invalid neighbors
                if (nbr_id == RouterTraitsT::invalid_vertex_id) { continue; }
                // Skip already visited neighbors
                if (visited_table.test(nbr_id)) { continue; }
                // Mark as visited
                visited_table.set(nbr_id);
                // Compute distance to neighbor
                const distance_t nbr_dist = this->_dist_func(query_vec, this->_vecs_data.get(nbr_id));
                // Try to add neighbor to candidate queue
                candidate_queue.try_push(nbr_id, nbr_dist);
            }
        }

        // Extract top-k result IDs from candidate queue
        return candidate_queue.extract_result_ids(this->_topk);
    }

private:
    /**
     * @brief Warm up the random sequence generator by triggering thread-local MKL stream creation.
     *
     * Forces all TBB worker threads to initialize their thread-local MKL random streams.
     * This ensures the first real query doesn't pay the initialization cost.
     */
    __attribute__((always_inline))
    auto _warmup_random_seq() -> void {
        const int num_threads = tbb_max_num_threads();
        tbb::parallel_for(
            tbb::blocked_range<int>(0, num_threads, 1),
            [&](const tbb::blocked_range<int>&) {
                // Trigger thread-local MKL stream initialization
                std::vector<vertex_id_t> dummy(1);
                _random_seq.generate(dummy, 1);
            }
        );
    }

    /** @brief Reference to the flat search graph for neighbor access. */
    const flat_search_graph_t& _flat_search_graph;

    /** @brief Candidate queue size for beam search. */
    vertex_num_t _candidate_queue_size = 0;

    /** @brief Pool of thread-local visited bitmaps for parallel beam search. */
    mutable visited_table_pool_t _visited_table_pool;

    /** @brief Thread-safe random sequence generator (internally uses thread-local MKL streams). */
    mutable random_seq_t _random_seq;

};  // class MonolayerGraphRouter

}   // namespace cpu
}   // namespace artea