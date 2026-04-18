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
 * @FilePath: /Artea/include/artea/cpu/router/compact_refining_graph_router.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: compact_mode specialization of RefiningGraphRouter.
 *               Operates on CompactRefiningGraph (CSR vertex_id_t neighbors).
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
#include <artea/cpu/router/data_structures/candidate_queue_concept.hpp>
#include <artea/cpu/router/visited_table_concept.hpp>
#include <artea/cpu/router/detail/beam_loop.hpp>
#include <artea/cpu/router/detail/compact_flat_range.hpp>

namespace artea {
namespace cpu {
namespace compact {

template <typename RouterTraitsT>
class RefiningGraphRouter :
    public RouterTraitsT::template vector_router_t<RefiningGraphRouter<RouterTraitsT>>
{

    using vertex_num_t = typename RouterTraitsT::vertex_num_t;
    using vertex_id_t = typename RouterTraitsT::vertex_id_t;
    using vec_ele_t = typename RouterTraitsT::vec_ele_t;
    using distance_t = typename RouterTraitsT::distance_t;
    using dist_func_t = typename RouterTraitsT::dist_func_t;
    using vector_array_t = typename RouterTraitsT::vector_array_t;
    using query_vecs_t = typename RouterTraitsT::query_vecs_t;
    using compact = typename RouterTraitsT::compact;
    using random_seq_t = typename RouterTraitsT::random_seq_t;
    using candidate_queue_t = typename RouterTraitsT::candidate_queue_t;
    using visited_table_t = typename RouterTraitsT::visited_table_t;
    using visited_table_pool_t = typename RouterTraitsT::visited_table_pool_t;
    using knn_results_t = typename RouterTraitsT::knn_results_t;
    using base_class_t = typename RouterTraitsT::template vector_router_t<RefiningGraphRouter<RouterTraitsT>>;

public:

    RefiningGraphRouter(
        const vector_array_t& vecs_data,
        const dist_func_t& dist_func,
        const compact::refining_graph_t& compact_refining_graph,
        const uint32_t topk,
        const vertex_num_t candidate_queue_size = 16
    ) : base_class_t(vecs_data, dist_func, topk),
        _compact_refining_graph(compact_refining_graph),
        _candidate_queue_size(candidate_queue_size),
        _visited_table_pool(vecs_data.get_num_vecs()),
        _random_seq()
    {
        if (candidate_queue_size < topk) {
            ARTEA_ERROR(fmt::format(
                "candidate_queue_size ({}) must be >= topk ({})",
                candidate_queue_size, topk
            ));
        }
    }

    auto initialize(bool with_entry_point = false) -> void {
        _visited_table_pool.warmup();
        if (!with_entry_point) { _warmup_random_seq(); }
    }

    /**
     * @brief Query the top-k nearest vertices using proximity graph.
     * @param query_vec Pointer to the query vector data.
     * @return knn_results_t Flat array of topk result entries sorted by distance.
     */
    __attribute__((always_inline))
    auto query(const vec_ele_t* query_vec) const -> knn_results_t {
        auto& visited_table = _visited_table_pool.acquire();
        return _beam_search(query_vec, visited_table, _random_seq);
    }

    /**
     * @brief Query the top-k nearest vertices using proximity graph with an entry point.
     * @param query_vec Pointer to the query vector data.
     * @param entry_point Starting vertex ID for the search.
     * @return knn_results_t Flat array of topk result entries sorted by distance.
     */
    __attribute__((always_inline))
    auto query(const vec_ele_t* query_vec, const vertex_id_t entry_point) const -> knn_results_t {
        auto& visited_table = _visited_table_pool.acquire();
        return _beam_search(query_vec, visited_table, entry_point);
    }

    /**
     * @brief Perform batch queries to find the top-k nearest vertices for multiple vectors.
     *
     * This implementation always parallelizes the batch processing (Inter-query parallelism) using TBB.
     *
     * @param query_vecs A VectorArray containing the query vectors.
     * @return knn_results_t Flat array of num_queries * topk result entries in row-major order.
     */
    auto batch_query(const query_vecs_t& query_vecs) const -> knn_results_t {
        const vertex_num_t num_queries = query_vecs.get_num_vecs();
        const uint32_t K = this->_topk;

        // Pre-allocate flat result array (num_queries * K entries)
        knn_results_t results(num_queries * K);

        tbb::parallel_for(
            tbb::blocked_range<vertex_num_t>(0, num_queries),
            [&](const tbb::blocked_range<vertex_num_t>& r) {
                // acquire per-query via this->query() internally; no
                // outer acquire needed since each inner query() already
                // refreshes the visited table.
                for (vertex_num_t i = r.begin(); i != r.end(); ++i) {
                    const vec_ele_t* q_vec = query_vecs.get(i);
                    auto topk_results = this->query(q_vec);
                    std::copy(topk_results.begin(), topk_results.end(), results.begin() + i * K);
                }
            }
        );

        return results;
    }

    /**
     * @brief Perform batch queries with a shared entry point.
     *
     * This implementation always parallelizes the batch processing (Inter-query parallelism) using TBB.
     *
     * @param query_vecs A VectorArray containing the query vectors.
     * @param entry_point Shared entry point vertex ID for all queries.
     * @return knn_results_t Flat array of num_queries * topk result entries in row-major order.
     */
    auto batch_query(const query_vecs_t& query_vecs, const vertex_id_t entry_point) const -> knn_results_t {
        const vertex_num_t num_queries = query_vecs.get_num_vecs();
        const uint32_t K = this->_topk;

        // Pre-allocate flat result array (num_queries * K entries)
        knn_results_t results(num_queries * K);

        tbb::parallel_for(
            tbb::blocked_range<vertex_num_t>(0, num_queries),
            [&](const tbb::blocked_range<vertex_num_t>& r) {
                for (vertex_num_t i = r.begin(); i != r.end(); ++i) {
                    // acquire() clears each time, so move it inside the
                    // loop to get a fresh table per query.
                    auto& visited = _visited_table_pool.acquire();
                    const vec_ele_t* q_vec = query_vecs.get(i);
                    auto topk_results = _beam_search(q_vec, visited, entry_point);
                    std::copy(topk_results.begin(), topk_results.end(), results.begin() + i * K);
                }
            }
        );

        return results;
    }

private:
    /**
     * @brief Perform beam search on the proximity graph to find top-k nearest neighbors.
     * @param query_vec Pointer to the query vector data.
     * @param visited_table Reference to the visited table for tracking explored vertices.
     * @param entry_point Starting vertex ID for the search.
     * @return Vector of vertex IDs representing the top-k nearest neighbors.
     */
    auto _beam_search(
        const vec_ele_t* query_vec,
        visited_table_t& visited_table,
        const vertex_id_t entry_point
    ) const -> knn_results_t {

        // Initialize candidate queue with capacity = max(topk, _candidate_queue_size)
        const vertex_num_t queue_capacity = std::max(this->_topk, _candidate_queue_size);
        candidate_queue_t candidate_queue(queue_capacity);

        // Compute distance to entry point and initialize queue
        const distance_t entry_dist = this->_dist_func(query_vec, this->_vecs_data.get(entry_point));

        candidate_queue.try_push(entry_point, entry_dist);
        visited_table.set(entry_point);

        const detail::CompactFlatRange<typename RouterTraitsT::compact::refining_graph_t>
            nbrs_range(_compact_refining_graph);
        detail::beam_loop_body<RouterTraitsT>(
            query_vec, nbrs_range, candidate_queue,
            visited_table, this->_dist_func, this->_vecs_data);

        // Extract top-k result IDs from candidate queue
        return candidate_queue.extract_results(this->_topk);
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
    auto _beam_search(
        const vec_ele_t* query_vec,
        visited_table_t& visited_table,
        random_seq_t& random_seq
    ) const -> knn_results_t {

        // Initialize candidate queue with capacity = max(topk, _candidate_queue_size)
        const vertex_num_t queue_capacity = std::max(this->_topk, _candidate_queue_size);
        candidate_queue_t candidate_queue(queue_capacity);

        // Initialize candidate queue with random vertices
        // (random_initialize → seeded_initialize already marks every seed
        // as visited, so the beam loop body can start without re-marking.)
        candidate_queue.random_initialize(
            random_seq,
            this->_dist_func,
            query_vec,
            this->_vecs_data,
            visited_table
        );

        const detail::CompactFlatRange<typename RouterTraitsT::compact::refining_graph_t>
            nbrs_range(_compact_refining_graph);
        detail::beam_loop_body<RouterTraitsT>(
            query_vec, nbrs_range, candidate_queue,
            visited_table, this->_dist_func, this->_vecs_data);

        // Extract top-k result IDs from candidate queue
        return candidate_queue.extract_results(this->_topk);
    }

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
                _random_seq.generate(dummy, /*upper_bound=*/1, /*num=*/1);
            }
        );
    }

    /** @brief Reference to the flat search graph for neighbor access. */
    const compact::refining_graph_t& _compact_refining_graph;

    /** @brief Candidate queue size for beam search. */
    vertex_num_t _candidate_queue_size = 0;

    /** @brief Pool of thread-local visited bitmaps for parallel beam search. */
    mutable visited_table_pool_t _visited_table_pool;

    /** @brief Thread-safe random sequence generator (internally uses thread-local MKL streams). */
    mutable random_seq_t _random_seq;

};  // class RefiningGraphRouter

}   // namespace compact
}   // namespace cpu
}   // namespace artea
