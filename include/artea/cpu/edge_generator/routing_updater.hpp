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
 * @FilePath: /Artea/include/artea/cpu/edge_generator/routing_updater.hpp
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
 * @brief RoutingUpdater uses a construct-mode MonolayerGraphRouter to find
 *        candidate nearest neighbors for each pivot vertex and logs them.
 *
 * @tparam GraphFactoryTraitsT Must expose both EdgeGeneratorTraits and RouterTraits
 *         (i.e. GraphFactoryTraits or any traits that inherits both).
 */
template <typename EdgeGeneratorTraitsT>
class RoutingUpdater :
    public EdgeGeneratorTraitsT::template neighbor_updater_t<RoutingUpdater<EdgeGeneratorTraitsT>>
{
    using vertex_id_t = typename EdgeGeneratorTraitsT::vertex_id_t;
    using vertex_num_t = typename EdgeGeneratorTraitsT::vertex_num_t;
    using vec_ele_t = typename EdgeGeneratorTraitsT::vec_ele_t;
    using distance_t = typename EdgeGeneratorTraitsT::distance_t;
    using vector_array_t = typename EdgeGeneratorTraitsT::vector_array_t;
    using nbr_t = typename EdgeGeneratorTraitsT::nbr_t;
    using nbr_arr_t = typename EdgeGeneratorTraitsT::nbr_arr_t;
    using log_table_t = typename EdgeGeneratorTraitsT::log_table_t;
    using dist_func_t = typename EdgeGeneratorTraitsT::dist_func_t;
    using flat_graph_t = typename EdgeGeneratorTraitsT::flat_graph_t;
    using graph_mode_t = typename EdgeGeneratorTraitsT::graph_mode_t;
    using router_t = typename EdgeGeneratorTraitsT::template monolayer_graph_router_t<graph_mode_t::construct_mode>;
    using base_class_t = typename EdgeGeneratorTraitsT::template neighbor_updater_t<RoutingUpdater<EdgeGeneratorTraitsT>>;

public:
    static constexpr const char* updater_name = "routing_updater";

    /**
     * @brief Constructor.
     * @param dist_func            Distance function reference.
     * @param vecs_data            Vector array containing all vertex data.
     * @param log_table            Log table for recording edge operations.
     * @param flat_graph           The flat graph to navigate during construction.
     * @param topk                 Number of nearest neighbors to retrieve per query.
     * @param candidate_queue_size Beam width for the router's candidate queue.
     */
    RoutingUpdater(
        const dist_func_t& dist_func,
        const vector_array_t& vecs_data,
        log_table_t& log_table,
        const flat_graph_t& flat_graph,
        const vertex_num_t topk,
        const vertex_num_t candidate_queue_size
    ) : base_class_t(dist_func, vecs_data, log_table),
        _router(vecs_data, dist_func, flat_graph, topk, candidate_queue_size),
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
        const vertex_id_t pivot_vid,
        nbr_arr_t& origin_nbrs
    ) -> void {
        const vec_ele_t* pivot_vec = this->_vecs_data.get(pivot_vid);
        auto knn_results = _router.query(pivot_vec);

        std::vector<vertex_id_t> knn_ids;
        std::vector<distance_t> knn_dists;
        knn_ids.reserve(knn_results.size());
        knn_dists.reserve(knn_results.size());
        for (const auto& entry : knn_results) {
            knn_ids.push_back(entry.get_id());
            knn_dists.push_back(entry.get_distance());
        }

        this->_log_table.write_logs(pivot_vid, knn_ids, knn_dists);
    }

private:
    /** @brief Construct-mode router used to find approximate NNs. */
    router_t _router;

    /** @brief Number of nearest neighbors to retrieve per query. */
    const vertex_num_t _topk;

};  // class RoutingUpdater

}   // namespace cpu
}   // namespace artea
