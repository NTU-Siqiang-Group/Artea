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
 * @FilePath: /Artea/include/artea/cpu/edge_generator/reverse_updater.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Reverse edge updater for building bidirectional graphs.
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace artea {
namespace cpu {

template <typename EdgeGeneratorTraitsT>
class ReverseUpdater :
    public EdgeGeneratorTraitsT::template neighbor_updater_t<ReverseUpdater<EdgeGeneratorTraitsT>> {

    using vertex_id_t = typename EdgeGeneratorTraitsT::vertex_id_t;
    using vertex_num_t = typename EdgeGeneratorTraitsT::vertex_num_t;
    using vec_ele_t = typename EdgeGeneratorTraitsT::vec_ele_t;
    using distance_t = typename EdgeGeneratorTraitsT::distance_t;
    using ratio_t = typename EdgeGeneratorTraitsT::ratio_t;
    using vector_array_t = typename EdgeGeneratorTraitsT::vector_array_t;
    using nbr_t = typename EdgeGeneratorTraitsT::nbr_t;
    using nbr_arr_t = typename EdgeGeneratorTraitsT::nbr_arr_t;
    using log_table_t = typename EdgeGeneratorTraitsT::log_table_t;
    using dist_func_t = typename EdgeGeneratorTraitsT::dist_func_t;
    using base_class_t = typename EdgeGeneratorTraitsT::template neighbor_updater_t<ReverseUpdater<EdgeGeneratorTraitsT>>;

public:
    /**
     * @brief Constructor for ReverseUpdater.
     * @param dist_func Distance function reference.
     * @param vecs_arr Vector array containing all vertex data.
     * @param log_table Log table for recording edge operations.
     */
    ReverseUpdater(
        const dist_func_t& dist_func,
        const vector_array_t& vecs_arr,
        log_table_t& log_table
    ) : base_class_t(dist_func, vecs_arr, log_table) {}

    /**
     * @brief Add reverse edges for all neighbors in origin_nbrs.
     *
     * For each neighbor nbr in origin_nbrs of pivot_vid, this operator adds a
     * reverse edge from nbr to pivot_vid with the same distance as the forward edge.
     * This creates bidirectional edges, allowing navigation in both directions.
     *
     * @param pivot_vid The vertex whose outgoing edges we're processing.
     * @param origin_nbrs The neighbor array of pivot_vid (not modified).
     */
    auto operator()(
        const vertex_id_t pivot_vid,
        nbr_arr_t& origin_nbrs
    ) -> void {
        // For each neighbor in origin_nbrs, add a reverse edge from that neighbor to pivot_vid
        for (vertex_num_t i = 0; i < origin_nbrs.size(); ++i) {
            const nbr_t& nbr = origin_nbrs[i];
            vertex_id_t nbr_id = nbr.get_id();
            distance_t dist = nbr.get_distance();

            // Add reverse edge: from nbr_id to pivot_vid with the same distance
            // This effectively adds pivot_vid as an incoming edge to nbr_id
            this->_log_table.write_log(
                /* executor_vid = */nbr_id,
                /* nbr_id = */pivot_vid,
                /* new_edge_dist = */dist
            );
        }

        // Note: origin_nbrs is not modified, as we only add reverse edges via log table
    }

};  // class ReverseUpdater

}   // namespace cpu
}   // namespace artea