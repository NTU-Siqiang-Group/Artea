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
 * @FilePath: /Artea/include/artea/cpu/refiner/updaters/routing_updater.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Routing-based edge updater: uses a construct-mode router to find
 *               approximate nearest neighbors and writes them to the log table.
 */

#pragma once

#include <cstddef>
#include <vector>

namespace artea {
namespace cpu {

/**
 * @brief RoutingUpdater uses a SingleLayerRouter to find candidate nearest
 *        neighbors for each pivot vertex and logs them.
 *
 * @tparam RefinerTraitsT The refiner traits type.
 */
template <typename RefinerTraitsT>
class RoutingUpdater :
    public RefinerTraitsT::template neighbor_updater_t<RoutingUpdater<RefinerTraitsT>>
{
    using vertex_id_t = typename RefinerTraitsT::vertex_id_t;
    using vertex_num_t = typename RefinerTraitsT::vertex_num_t;
    using vec_ele_t = typename RefinerTraitsT::vec_ele_t;
    using distance_t = typename RefinerTraitsT::distance_t;
    using vector_array_t = typename RefinerTraitsT::vector_array_t;
    using nbr_t = typename RefinerTraitsT::nbr_t;
    using nbr_arr_t = typename RefinerTraitsT::nbr_arr_t;
    using log_table_t = typename RefinerTraitsT::log_table_t;
    using dist_func_t = typename RefinerTraitsT::dist_func_t;
    using router_t = typename RefinerTraitsT::single_layer_router_t;
    using refining_graph_t = typename RefinerTraitsT::dynamic::refining_graph_t;
    using base_class_t = typename RefinerTraitsT::template neighbor_updater_t<RoutingUpdater<RefinerTraitsT>>;

public:
    static constexpr const char* updater_name = "routing_updater";

    /**
     * @brief Constructor.
     * @param dist_func            Distance function reference.
     * @param vecs_data            Vector array containing all vertex data.
     * @param log_table            Log table for recording edge operations.
     * @param refining_graph       The descent graph the router beam-searches over.
     * @param topk                 Number of nearest neighbors to retrieve per query.
     * @param candidate_queue_size Beam width for the router's candidate queue.
     */
    RoutingUpdater(
        const dist_func_t&        dist_func,
        const vector_array_t&     vecs_data,
        log_table_t&              log_table,
        const refining_graph_t&   refining_graph,
        const vertex_num_t        topk,
        const vertex_num_t        candidate_queue_size
    ) : base_class_t(dist_func, vecs_data, log_table, refining_graph),
        _router(vecs_data, dist_func, topk, candidate_queue_size),
        _topk(topk)
    {   _router.initialize();   }

    /**
     * @brief For each pivot vertex, search for its approximate NNs via the router
     *        and write the results to the log table.
     *
     * @param pivot_vid    The vertex being processed.
     * @param origin_nbrs  Unused; present only to satisfy NeighborUpdater interface.
     */
    auto update_impl(
        const vertex_id_t layer_vid,
        nbr_arr_t& origin_nbrs
    ) -> void {
        const vertex_id_t storage_vid = this->_refining_graph.get_storage_vid(layer_vid);
        const vec_ele_t* pivot_vec = this->_vecs_data.get(storage_vid);
        // Warm-start beam search with the pivot's own (sorted, distance-
        // cached) neighbor slot so the first few iterations don't have
        // to re-explore from scratch.
        const nbr_arr_t& pivot_nbrs = this->_refining_graph.fetch_nbrs(storage_vid);
        auto knn_results = _router.query(pivot_vec, this->_refining_graph, pivot_nbrs);

        std::vector<vertex_id_t> knn_ids;
        std::vector<distance_t> knn_dists;
        knn_ids.reserve(knn_results.size());
        knn_dists.reserve(knn_results.size());
        const vertex_num_t max_sz = this->_refining_graph.layer_config().max_nbr_size();
        for (const auto& entry : knn_results) {
            if (entry.is_invalid()) { continue; }
            if (entry.get_vid() == storage_vid) { continue; }
            const nbr_arr_t& target_nbrs = this->_refining_graph.fetch_nbrs(storage_vid);
            if (target_nbrs.size() >= max_sz &&
                target_nbrs[max_sz - 1].get_distance() <= entry.get_distance()) {
                continue;
            }
            knn_ids.push_back(entry.get_vid());
            knn_dists.push_back(entry.get_distance());
        }

        this->_log_table.write_logs(layer_vid, knn_ids, knn_dists);
    }

private:
    /** @brief Construct-mode router used to find approximate NNs. */
    router_t _router;

    /** @brief Number of nearest neighbors to retrieve per query. */
    const vertex_num_t _topk;

};  // class RoutingUpdater

}   // namespace cpu
}   // namespace artea
