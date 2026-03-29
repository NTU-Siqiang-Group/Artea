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
 * @FilePath: /Artea/include/artea/cpu/router/construct_mode_hierarchical_graph_router.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: construct_mode specialization of HierarchicalGraphRouter.
 *               Operates on HierarchicalGraph (nbr_t neighbors) for build-time navigation.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include <artea/cpu/framework/type_traits/router_traits.hpp>
#include <artea/cpu/router/candidate_queue_concept.hpp>
#include <artea/cpu/router/visited_table_concept.hpp>

namespace artea {
namespace cpu {

template <typename RouterTraitsT>
class HierarchicalGraphRouter<RouterTraitsT, GraphModeT::construct_mode> :
    public RouterTraitsT::template vector_router_t<HierarchicalGraphRouter<RouterTraitsT, GraphModeT::construct_mode>>
{
    using vertex_num_t = typename RouterTraitsT::vertex_num_t;
    using vertex_id_t = typename RouterTraitsT::vertex_id_t;
    using layer_id_t = typename RouterTraitsT::layer_id_t;
    using layer_num_t = typename RouterTraitsT::layer_num_t;
    using vec_ele_t = typename RouterTraitsT::vec_ele_t;
    using distance_t = typename RouterTraitsT::distance_t;
    using dist_func_t = typename RouterTraitsT::dist_func_t;
    using vector_array_t = typename RouterTraitsT::vector_array_t;
    using query_vecs_t = typename RouterTraitsT::query_vecs_t;
    using conv_graph_index_t = typename RouterTraitsT::conv_graph_index_t;
    using artea_graph_index_t = typename RouterTraitsT::artea_graph_index_t;
    using nbr_arr_t = typename RouterTraitsT::nbr_arr_t;
    using candidate_queue_t = typename RouterTraitsT::candidate_queue_t;
    using visited_table_t = typename RouterTraitsT::visited_table_t;
    using visited_table_pool_t = typename RouterTraitsT::visited_table_pool_t;
    using knn_results_t = typename RouterTraitsT::knn_results_t;
    using base_class_t = typename RouterTraitsT::template vector_router_t<HierarchicalGraphRouter<RouterTraitsT, GraphModeT::construct_mode>>;

public:

    HierarchicalGraphRouter(
        const vector_array_t& vecs_data,
        const dist_func_t& dist_func,
        const artea_graph_index_t& hierarchical_graph,
        const uint32_t topk,
        const vertex_num_t candidate_queue_size,
        const vertex_num_t ul_extracted_nbr_size = 32,
        const vertex_num_t bl_extracted_nbr_size = 64
    ) : base_class_t(vecs_data, dist_func, topk),
        _hierarchical_graph(hierarchical_graph),
        _candidate_queue_size(candidate_queue_size),
        _ul_extracted_nbr_size(ul_extracted_nbr_size),
        _bl_extracted_nbr_size(bl_extracted_nbr_size),
        _visited_table_pool(vecs_data.get_num_vecs())
    {
        if (candidate_queue_size < topk) {
            ARTEA_ERROR(fmt::format(
                "candidate_queue_size ({}) must be >= topk ({})",
                candidate_queue_size, topk
            ));
        }
    }

    auto initialize_impl() -> void {
        _visited_table_pool.warmup();
    }

    /**
     * @brief Query the top-k nearest vertices using hierarchical graph (build-time).
     * @param query_vec Pointer to the query vector data.
     * @return knn_results_t Flat array of topk result entries sorted by distance.
     */
    __attribute__((always_inline))
    auto query_impl(const vec_ele_t* query_vec) const -> knn_results_t {
        auto& visited_table = _visited_table_pool.acquire();
        return _hierarchical_search(query_vec, visited_table);
    }

    /**
     * @brief Perform batch queries to find the top-k nearest vertices for multiple vectors.
     * @param query_vecs A VectorArray containing the query vectors.
     * @return knn_results_t Flat array of num_queries * topk result entries in row-major order.
     */
    auto batch_query_impl(const query_vecs_t& query_vecs) const -> knn_results_t {
        const vertex_num_t num_queries = query_vecs.get_num_vecs();
        const uint32_t K = this->_topk;
        knn_results_t results(num_queries * K);

        tbb::parallel_for(
            tbb::blocked_range<vertex_num_t>(0, num_queries),
            [&](const tbb::blocked_range<vertex_num_t>& r) {
                auto& visited = _visited_table_pool.acquire();
                for (vertex_num_t i = r.begin(); i != r.end(); ++i) {
                    const vec_ele_t* q_vec = query_vecs.get(i);
                    auto topk_results = _hierarchical_search(q_vec, visited);
                    std::copy(topk_results.begin(), topk_results.end(), results.begin() + i * K);
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
     * @return knn_results_t Flat array of topk result entries sorted by distance.
     */
    auto _hierarchical_search(
        const vec_ele_t* query_vec,
        visited_table_t& visited_table
    ) const -> knn_results_t {
        const layer_num_t num_layers = _hierarchical_graph.get_num_layers();
        const layer_id_t top_layer_id = num_layers - 1;

        vertex_id_t current_nearest = _hierarchical_graph.get_entry_point();
        const auto& top_layer_vecs = _hierarchical_graph.get_hier_vecs_manager().get_layer_vecs(top_layer_id);
        distance_t current_dist = this->_dist_func(query_vec, top_layer_vecs.get(current_nearest));

        // Greedy descent from top layer down to layer 1
        for (layer_id_t layer_id = top_layer_id; layer_id > 0; --layer_id) {
            _greedy_search_layer(query_vec, layer_id, current_nearest, current_dist);

            const auto& inter_layer_links = _hierarchical_graph.get_inter_layer_links();
            current_nearest = inter_layer_links.get_linked_vertex(layer_id, current_nearest);
        }

        // Beam search on bottom layer (layer 0)
        candidate_queue_t candidate_queue(_candidate_queue_size);
        candidate_queue.try_push(current_nearest, current_dist);
        visited_table.set(current_nearest);

        _beam_search_layer(query_vec, 0, visited_table, candidate_queue);

        return candidate_queue.extract_results(this->_topk);
    }

    /**
     * @brief Perform greedy search on a single upper layer.
     * @param query_vec Pointer to the query vector data.
     * @param layer_id Current layer ID.
     * @param current_nearest Reference to current nearest vertex (modified in-place).
     * @param current_dist Reference to current nearest distance (modified in-place).
     */
    auto _greedy_search_layer(
        const vec_ele_t* query_vec,
        const layer_id_t layer_id,
        vertex_id_t& current_nearest,
        distance_t& current_dist
    ) const -> void {
        const conv_graph_index_t& layer_graph = _hierarchical_graph.get_layer_graph(layer_id);
        const auto& layer_vecs = _hierarchical_graph.get_hier_vecs_manager().get_layer_vecs(layer_id);

        bool improved = true;
        while (improved) {
            improved = false;
            const nbr_arr_t& nbrs = layer_graph.fetch_nbrs(current_nearest);
            const vertex_num_t nbr_limit = std::min(static_cast<vertex_num_t>(nbrs.size()), _ul_extracted_nbr_size);
            for (vertex_num_t i = 0; i < nbr_limit; ++i) {
                const vertex_id_t nbr_id = nbrs[i].get_id();
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
     * @param layer_id Current layer ID.
     * @param visited_table Reference to the visited table.
     * @param candidate_queue Reference to the candidate queue (modified in-place).
     */
    auto _beam_search_layer(
        const vec_ele_t* query_vec,
        const layer_id_t layer_id,
        visited_table_t& visited_table,
        candidate_queue_t& candidate_queue
    ) const -> void {
        const conv_graph_index_t& layer_graph = _hierarchical_graph.get_layer_graph(layer_id);
        const auto& layer_vecs = _hierarchical_graph.get_hier_vecs_manager().get_layer_vecs(layer_id);

        while (!candidate_queue.empty()) {
            if (candidate_queue.should_terminate()) { break; }
            auto [current_id, current_dist] = candidate_queue.pop_best_unexplored();
            if (current_id == RouterTraitsT::invalid_vertex_id) { break; }

            const nbr_arr_t& nbrs = layer_graph.fetch_nbrs(current_id);
            const vertex_num_t nbr_limit = std::min(static_cast<vertex_num_t>(nbrs.size()), _bl_extracted_nbr_size);
            for (vertex_num_t i = 0; i < nbr_limit; ++i) {
                const vertex_id_t nbr_id = nbrs[i].get_id();
                if (nbr_id == RouterTraitsT::invalid_vertex_id) { break; }
                if (visited_table.test(nbr_id)) { continue; }
                visited_table.set(nbr_id);
                const distance_t nbr_dist = this->_dist_func(query_vec, layer_vecs.get(nbr_id));
                candidate_queue.try_push(nbr_id, nbr_dist);
            }
        }
    }

    /** @brief Reference to the hierarchical graph (build-time, nbr_t neighbors). */
    const artea_graph_index_t& _hierarchical_graph;

    /** @brief Candidate queue size for bottom layer beam search. */
    vertex_num_t _candidate_queue_size;

    /** @brief Maximum number of neighbors to explore per vertex in upper layers. */
    vertex_num_t _ul_extracted_nbr_size = 32;

    /** @brief Maximum number of neighbors to explore per vertex in bottom layer. */
    vertex_num_t _bl_extracted_nbr_size = 64;

    /** @brief Pool of thread-local visited bitmaps. */
    mutable visited_table_pool_t _visited_table_pool;

};  // class HierarchicalGraphRouter<RouterTraitsT, GraphModeT::construct_mode>

}   // namespace cpu
}   // namespace artea
