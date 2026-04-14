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
 * @FilePath: /Artea/include/artea/cpu/refiner/random_updater.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Random edge updater for generating random neighbors.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <artea/cpu/utils/random_seq.hpp>

namespace artea {
namespace cpu {

template <typename RefinerTraitsT>
class RandomUpdater :
    public RefinerTraitsT::template neighbor_updater_t<RandomUpdater<RefinerTraitsT>>
{

    using vertex_id_t = typename RefinerTraitsT::vertex_id_t;
    using vertex_num_t = typename RefinerTraitsT::vertex_num_t;
    using vec_ele_t = typename RefinerTraitsT::vec_ele_t;
    using distance_t = typename RefinerTraitsT::distance_t;
    using ratio_t = typename RefinerTraitsT::ratio_t;
    using vector_array_t = typename RefinerTraitsT::vector_array_t;
    using nbr_t = typename RefinerTraitsT::nbr_t;
    using nbr_arr_t = typename RefinerTraitsT::nbr_arr_t;
    using log_table_t = typename RefinerTraitsT::log_table_t;
    using dist_func_t = typename RefinerTraitsT::dist_func_t;
    using refining_graph_t = typename RefinerTraitsT::dynamic::refining_graph_t;
    using base_class_t = typename RefinerTraitsT::template neighbor_updater_t<RandomUpdater<RefinerTraitsT>>;
    using random_seq_t = typename RefinerTraitsT::random_seq_t;

public:
    static constexpr const char* updater_name = "random_updater";

    /**
     * @brief Constructor for RandomUpdater.
     */
    RandomUpdater(
        const dist_func_t&        dist_func,
        const vector_array_t&     vecs_data,
        log_table_t&              log_table,
        const refining_graph_t&   refining_graph,
        const vertex_num_t        num_vertices,
        const vertex_num_t        rand_gen_size
    ) : base_class_t(dist_func, vecs_data, log_table, refining_graph),
        _num_vertices(num_vertices),
        _rand_gen_size(rand_gen_size),
        _random_seq() {}

    /**
     * @brief Generate random neighbors for the pivot vertex.
     *
     * This operator generates rand_gen_size random vertex IDs and writes them
     * to the log table as potential neighbors for the pivot vertex. The distances
     * are computed using the distance function.
     *
     * @param pivot_vid The vertex for which to generate random neighbors.
     * @param origin_nbrs The neighbor array of pivot_vid (not used in this updater).
     *
     * @note This operator is thread-safe and can be called in parallel by the
     *       propagate engine, as RandomSeq uses thread-local storage for random
     *       number generation.
     */
    auto update_impl(
        const vertex_id_t local_vid,
        const vertex_id_t global_vid,
        nbr_arr_t& origin_nbrs
    ) -> void {
        // Random ids are drawn in [0, _num_vertices) = local-row space;
        // translate each to a global vid for distance evaluation.
        std::vector<vertex_id_t> rand_ids_buffer(_rand_gen_size);
        _random_seq.generate(rand_ids_buffer, _num_vertices, _rand_gen_size);

        const vec_ele_t* pivot_vec = this->_vecs_data.get(global_vid);
        const vertex_num_t max_sz = this->_refining_graph.layer_config().max_nbr_size();

        std::vector<vertex_id_t> nbr_ids;
        std::vector<distance_t> nbr_dists;
        nbr_ids.reserve(_rand_gen_size);
        nbr_dists.reserve(_rand_gen_size);

        for (vertex_num_t i = 0; i < _rand_gen_size; ++i) {
            const vertex_id_t rand_nbr_global = this->_refining_graph.vid_at(rand_ids_buffer[i]);
            if (rand_nbr_global == global_vid) { continue; }
            const distance_t dist = this->_dist_func(pivot_vec, this->_vecs_data.get(rand_nbr_global));

            const nbr_arr_t& pivot_nbrs = this->_refining_graph.fetch_nbrs(global_vid);
            if (pivot_nbrs.size() >= max_sz &&
                pivot_nbrs[max_sz - 1].get_distance() <= dist) {
                continue;
            }
            nbr_ids.push_back(rand_nbr_global);
            nbr_dists.push_back(dist);
        }

        this->_log_table.write_logs(local_vid, nbr_ids, nbr_dists);
    }

private:
    /** @brief Total number of vertices in the graph (upper bound for random ID generation). */
    const vertex_num_t _num_vertices;

    /** @brief Number of random neighbors to generate per vertex. */
    const vertex_num_t _rand_gen_size;

    /** @brief Thread-safe random sequence generator. */
    random_seq_t _random_seq;

};  // class RandomUpdater

}   // namespace cpu
}   // namespace artea

