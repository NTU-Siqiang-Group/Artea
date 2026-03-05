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
 * @FilePath: /Artea/include/artea/cpu/edge_generator/random_updater.hpp
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

template <typename EdgeGeneratorTraitsT>
class RandomUpdater :
    public EdgeGeneratorTraitsT::template neighbor_updater_t<RandomUpdater<EdgeGeneratorTraitsT>>
{

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
    using base_class_t = typename EdgeGeneratorTraitsT::template neighbor_updater_t<RandomUpdater<EdgeGeneratorTraitsT>>;
    using base_traits_t = typename EdgeGeneratorTraitsT::base_traits_t;

public:
    /**
     * @brief Constructor for RandomUpdater.
     * @param dist_func Distance function reference.
     * @param vecs_arr Vector array containing all vertex data.
     * @param log_table Log table for recording edge operations.
     * @param num_vertices Total number of vertices in the graph (used as upper bound for random ID generation).
     * @param rand_gen_size Number of random neighbors to generate for each vertex.
     */
    RandomUpdater(
        const dist_func_t& dist_func,
        const vector_array_t& vecs_arr,
        log_table_t& log_table,
        const vertex_num_t num_vertices,
        const vertex_num_t rand_gen_size
    ) : base_class_t(dist_func, vecs_arr, log_table),
        _rand_gen_size(rand_gen_size),
        _random_seq(num_vertices) {}

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
        const vertex_id_t pivot_vid,
        nbr_arr_t& origin_nbrs
    ) -> void {
        // Create a local buffer for storing generated random IDs
        std::vector<vertex_id_t> rand_ids_buffer(_rand_gen_size);

        // Generate rand_gen_size random vertex IDs
        _random_seq.generate(rand_ids_buffer, _rand_gen_size);

        // Get the pivot vertex vector
        const vec_ele_t* pivot_vec = this->_vecs_arr.get(pivot_vid);

        // For each random neighbor, compute distance and write to log table
        for (vertex_num_t i = 0; i < _rand_gen_size; ++i) {
            vertex_id_t rand_nbr_id = rand_ids_buffer[i];

            // Skip if the random ID is the pivot itself
            if (rand_nbr_id == pivot_vid) { continue; }

            // Get the random neighbor vector and compute distance
            const vec_ele_t* rand_nbr_vec = this->_vecs_arr.get(rand_nbr_id);
            distance_t rand_nbr_dist = this->_dist_func(pivot_vec, rand_nbr_vec);

            // Write the random edge to the log table
            this->_log_table.write_log(
                /* executor_vid = */pivot_vid,
                /* nbr_id = */rand_nbr_id,
                /* new_edge_dist = */rand_nbr_dist
            );
        }
    }

private:
    /** @brief Number of random neighbors to generate per vertex. */
    const vertex_num_t _rand_gen_size;

    /** @brief Thread-safe random sequence generator. */
    RandomSeq<base_traits_t> _random_seq;

};  // class RandomUpdater

}   // namespace cpu
}   // namespace artea

