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
 * @FilePath: /Artea/include/artea/cpu/router/hierarchical_graph_router.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Hierarchical graph router for HNSW-like search.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <memory>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include <artea/cpu/router/candidate_queue_concept.hpp>
#include <artea/cpu/router/visited_table_concept.hpp>

namespace artea {
namespace cpu {

template <
    typename RouterTraitsT,
    CandidateQueue CandidateQueueImpl = typename RouterTraitsT::std_candidate_queue_t,
    VisitedTable VisitedTableImpl = typename RouterTraitsT::thread_local_bitmap_t
>   requires CandidateQueue<CandidateQueueImpl> && VisitedTable<VisitedTableImpl>
class HierarchicalGraphRouter :
    public RouterTraitsT::template vector_router_t<HierarchicalGraphRouter<RouterTraitsT, CandidateQueueImpl, VisitedTableImpl>>
{

    using candidate_queue_t = CandidateQueueImpl;
    using visited_table_t = VisitedTableImpl;
    using visited_table_pool_t = typename RouterTraitsT::template visited_table_pool_t<VisitedTableImpl>;
    using vertex_num_t = typename RouterTraitsT::vertex_num_t;
    using vertex_id_t = typename RouterTraitsT::vertex_id_t;
    using layer_id_t = typename RouterTraitsT::layer_id_t;
    using layer_num_t = typename RouterTraitsT::layer_num_t;
    using vec_ele_t = typename RouterTraitsT::vec_ele_t;
    using distance_t = typename RouterTraitsT::distance_t;
    using dist_func_t = typename RouterTraitsT::dist_func_t;
    using vector_array_t = typename RouterTraitsT::vector_array_t;
    using query_vecs_t = typename RouterTraitsT::query_vecs_t;
    using idlist_array_t = typename RouterTraitsT::idlist_array_t;
    using flat_search_graph_t = typename RouterTraitsT::flat_search_graph_t;
    using hierarchical_search_graph_t = typename RouterTraitsT::hierarchical_search_graph_t;
    using inter_layer_links_t = typename RouterTraitsT::inter_layer_links_t;
    using base_class_t = typename RouterTraitsT::template vector_router_t<HierarchicalGraphRouter<RouterTraitsT, CandidateQueueImpl, VisitedTableImpl>>;

    static constexpr vertex_num_t min_num_layer_vertex = RouterTraitsT::min_num_layer_vertex;

public:
    HierarchicalGraphRouter(
        const vector_array_t& vecs_data,
        const dist_func_t& dist_func,
        const hierarchical_search_graph_t& hierarchical_search_graph,
        const uint32_t topk,
        const vertex_num_t candidate_queue_size
    ) : base_class_t(vecs_data, dist_func, topk),
        _hierarchical_search_graph(hierarchical_search_graph),
        _candidate_queue_size(candidate_queue_size),
        _visited_table_pool(vecs_data.get_num_vecs())
    {
        if (candidate_queue_size < topk) {
            logger.error(fmt::format(
                "candidate_queue_size ({}) must be >= topk ({})",
                candidate_queue_size, topk
            ));
        }
    }

    auto initialize_impl() -> void {
        _visited_table_pool.warmup();
    }

    /**
     * @brief Query the top-k nearest vertices using hierarchical graph.
     * @param query_vec Pointer to the query vector data.
     * @return std::vector<vertex_id_t> Vector containing the IDs of the top-k nearest vertices.
     */
    __attribute__((always_inline))
    auto query_impl(const vec_ele_t* query_vec) const -> std::vector<vertex_id_t> {
        auto& visited_table = _visited_table_pool.acquire();
        std::vector<vertex_id_t> result_ids = _hierarchical_search(query_vec, visited_table);
        return result_ids;
    }

    /**
     * @brief Perform batch queries to find the top-k nearest vertices for multiple vectors.
     * @param query_vecs A VectorArray containing the query vectors.
     * @return idlist_array_t Array with num_vecs=num_queries, dim=topk where each vector contains the top-k IDs for one query.
     */
    auto batch_query_impl(const query_vecs_t& query_vecs) const -> idlist_array_t {
        const vertex_num_t num_queries = query_vecs.get_num_vecs();
        idlist_array_t results(num_queries, this->_topk);

        tbb::parallel_for(
            tbb::blocked_range<vertex_num_t>(0, num_queries),
            [&](const tbb::blocked_range<vertex_num_t>& r) {
                auto& visited = _visited_table_pool.acquire();
                for (vertex_num_t i = r.begin(); i != r.end(); ++i) {
                    const vec_ele_t* q_vec = query_vecs.get(i);
                    auto topk_results = _hierarchical_search(q_vec, visited);
                    results.set(i, topk_results.data());
                    visited.clear();
                }
            }
        );

        return results;
    }

private:
    /**
     * @brief Perform hierarchical search from top layer to bottom layer.
     * @param query_vec Pointer to the query vector data.
     * @param visited_table Reference to the visited table for tracking explored vertices.
     * @return Vector of vertex IDs representing the top-k nearest neighbors.
     */
    auto _hierarchical_search(
        const vec_ele_t* query_vec,
        visited_table_t& visited_table
    ) const -> std::vector<vertex_id_t> {
        const layer_num_t num_layers = _hierarchical_search_graph.get_num_layers();
        const layer_id_t top_layer_id = num_layers - 1;

        // Initialize with entry point
        vertex_id_t current_nearest = _hierarchical_search_graph.get_entry_point();
        const auto& top_layer_vecs = _hierarchical_search_graph.get_hier_vecs_manager().get_layer_vecs(top_layer_id);
        distance_t current_dist = this->_dist_func(query_vec, top_layer_vecs.get(current_nearest));

        // Greedy search from top layer down to layer 1 (not including bottom layer 0)
        for (layer_id_t layer_id = top_layer_id; layer_id > 0; --layer_id) {
            // Greedy search on current layer
            _greedy_search_layer(
                query_vec,
                current_nearest,
                current_dist,
                layer_id
            );

            // Convert to next layer using inter-layer links
            const auto& inter_layer_links = _hierarchical_search_graph.get_inter_layer_links();
            current_nearest = inter_layer_links.get_linked_vertex(layer_id, current_nearest);
        }

        // Bottom layer search with candidate queue
        candidate_queue_t candidate_queue(_candidate_queue_size);
        candidate_queue.try_push(current_nearest, current_dist);
        visited_table.set(current_nearest);

        // Search bottom layer
        _beam_search_layer(
            query_vec,
            visited_table,
            candidate_queue,
            0
        );

        // Extract top-k results
        return candidate_queue.extract_result_ids(this->_topk);
    }

    /**
     * @brief Perform greedy search on a single upper layer.
     * @param query_vec Pointer to the query vector data.
     * @param current_nearest Reference to current nearest vertex (modified in-place).
     * @param current_dist Reference to current nearest distance (modified in-place).
     * @param layer_id Current layer ID.
     */
    auto _greedy_search_layer(
        const vec_ele_t* query_vec,
        vertex_id_t& current_nearest,
        distance_t& current_dist,
        const layer_id_t layer_id
    ) const -> void {
        const auto& layer_graph = _hierarchical_search_graph.get_layer_graph(layer_id);
        const auto& layer_vecs = _hierarchical_search_graph.get_hier_vecs_manager().get_layer_vecs(layer_id);

        bool improved = true;
        while (improved) {
            improved = false;
            const auto& neighbors = layer_graph.fetch_nbrs(current_nearest);

            for (const auto& nbr_id : neighbors) {
                if (nbr_id == RouterTraitsT::invalid_vertex_id) { continue; }

                const distance_t nbr_dist = this->_dist_func(query_vec, layer_vecs.get(nbr_id));

                if (nbr_dist < current_dist) {
                    current_nearest = nbr_id;
                    current_dist = nbr_dist;
                    improved = true;
                }
            }
        }
    }

    /**
     * @brief Perform beam search on a single layer.
     * @param query_vec Pointer to the query vector data.
     * @param visited_table Reference to the visited table.
     * @param candidate_queue Reference to the candidate queue (modified in-place).
     * @param layer_id Current layer ID.
     */
    auto _beam_search_layer(
        const vec_ele_t* query_vec,
        visited_table_t& visited_table,
        candidate_queue_t& candidate_queue,
        const layer_id_t layer_id
    ) const -> void {
        const auto& layer_graph = _hierarchical_search_graph.get_layer_graph(layer_id);
        // Get the layer-specific vectors for distance computation
        const auto& layer_vecs = _hierarchical_search_graph.get_hier_vecs_manager().get_layer_vecs(layer_id);

        // Beam search loop
        while (!candidate_queue.empty()) {
            // Termination check
            if (candidate_queue.should_terminate()) { break; }

            // Get the best unexplored candidate
            auto [current_id, current_dist] = candidate_queue.pop_best_unexplored();

            // Check for invalid entry
            if (current_id == RouterTraitsT::invalid_vertex_id) { break; }

            // Explore neighbors of current vertex
            const auto& neighbors = layer_graph.fetch_nbrs(current_id);

            for (const auto& nbr_id : neighbors) {
                // Skip invalid neighbors
                if (nbr_id == RouterTraitsT::invalid_vertex_id) { continue; }

                // Skip already visited neighbors
                if (visited_table.test(nbr_id)) { continue; }

                // Mark as visited
                visited_table.set(nbr_id);

                // Compute distance to neighbor using layer-specific vectors
                const distance_t nbr_dist = this->_dist_func(query_vec, layer_vecs.get(nbr_id));

                // Try to add neighbor to candidate queue
                candidate_queue.try_push(nbr_id, nbr_dist);
            }
        }
    }

    /**
     * @brief Convert candidate queue from current layer to next layer using inter-layer links.
     * @param candidate_queue Reference to the candidate queue (modified in-place).
     * @param current_layer_id Current layer ID.
     */
    auto _convert_to_next_layer(
        candidate_queue_t& candidate_queue,
        const layer_id_t current_layer_id
    ) const -> void {
        const auto& inter_layer_links = _hierarchical_search_graph.get_inter_layer_links();

        // Extract current results with distances
        auto current_results = candidate_queue.extract_results(candidate_queue.get_result_size());

        // Clear and rebuild with next layer vertex IDs
        candidate_queue.clear();

        for (const auto& [current_vid, dist] : current_results) {
            const vertex_id_t next_vid = inter_layer_links.get_linked_vertex(current_layer_id, current_vid);
            candidate_queue.try_push(next_vid, dist);
        }
    }

    /** @brief Reference to the hierarchical search graph. */
    const hierarchical_search_graph_t& _hierarchical_search_graph;

    /** @brief Candidate queue size for bottom layer. */
    vertex_num_t _candidate_queue_size;

    /** @brief Pool of thread-local visited bitmaps. */
    mutable visited_table_pool_t _visited_table_pool;

};  // class HierarchicalGraphRouter

}   // namespace cpu
}   // namespace artea
