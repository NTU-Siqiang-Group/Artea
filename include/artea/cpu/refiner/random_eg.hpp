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
 * @FilePath: /Artea/include/artea/cpu/refiner/random_eg.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Random edge generator for graph initialization.
 */

#pragma once

#include <algorithm>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

namespace artea {
namespace cpu {

/**
 * @brief Random edge generator for initializing graph with random neighbors.
 * @tparam RefinerTraitsT The refiner traits type.
 */
template <typename RefinerTraitsT>
class RandomEG {

    using vertex_num_t = typename RefinerTraitsT::vertex_num_t;
    using vertex_id_t = typename RefinerTraitsT::vertex_id_t;
    using vec_ele_t = typename RefinerTraitsT::vec_ele_t;
    using distance_t = typename RefinerTraitsT::distance_t;
    using vec_dim_t = typename RefinerTraitsT::vec_dim_t;
    using vector_array_t = typename RefinerTraitsT::vector_array_t;
    using dnbr_t = typename RefinerTraitsT::dnbr_t;
    using dnbr_arr_t = typename RefinerTraitsT::dnbr_arr_t;
    using dnbr_comp_t = typename RefinerTraitsT::dnbr_comp_t;
    using dist_func_t = typename RefinerTraitsT::dist_func_t;
    using random_seq_t = typename RefinerTraitsT::random_seq_t;

    static constexpr dnbr_comp_t nbr_comp {};

public:
    /**
     * @brief Construct a new RandomEG object.
     * @param dist_func Distance function for computing neighbor distances.
     */
    RandomEG(const dist_func_t& dist_func)
        : _dist_func(dist_func) {}

    /**
     * @brief Generate random edges for a descent graph.
     * @param descent_graph The graph to initialize with random edges.
     * @param init_nbr_size Number of random neighbors to generate for each vertex.
     */
    template <typename DescentGraphT>
    auto generate(
        DescentGraphT& descent_graph,
        const vertex_num_t init_nbr_size
    ) -> void {
        const vertex_num_t num_vertices = descent_graph.get_num_vertices();
        const vector_array_t& vecs_data = descent_graph.get_vecs_data();

        // RandomSeq uses thread-local storage internally, so it's safe to share across threads
        random_seq_t random_seq(num_vertices);

        tbb::parallel_for(
            tbb::blocked_range<vertex_id_t>(0, num_vertices),
            [&](const tbb::blocked_range<vertex_id_t>& r) {
                // Pre-allocate vector for random IDs (reused for each vertex in this thread)
                std::vector<vertex_id_t> random_nbr_ids(init_nbr_size);

                for (vertex_id_t vid = r.begin(); vid != r.end(); ++vid) {
                    dnbr_arr_t& nbrs = descent_graph.fetch_nbrs(vid);
                    const vec_ele_t* query_vec = vecs_data.get(vid);

                    // Generate random neighbor IDs
                    random_seq.generate(random_nbr_ids, init_nbr_size);

                    // Create neighbors with distances
                    nbrs.clear();
                    // Note: nbrs already has reserved capacity from DescentGraph constructor

                    for (vertex_num_t i = 0; i < init_nbr_size; ++i) {
                        const vertex_id_t nbr_id = random_nbr_ids[i];

                        // Skip self-loops
                        if (nbr_id == vid) {
                            continue;
                        }

                        const vec_ele_t* nbr_vec = vecs_data.get(nbr_id);
                        const distance_t dist = _dist_func(query_vec, nbr_vec);

                        // Create neighbor with status "new"
                        nbrs.emplace_back(nbr_id, dist, /* is_new = */ true);
                    }

                    // Sort by distance
                    std::sort(nbrs.begin(), nbrs.end(), nbr_comp);

                    // Remove duplicates (keep the one with smaller distance)
                    auto last = std::unique(
                        nbrs.begin(),
                        nbrs.end(),
                        [](const dnbr_t& a, const dnbr_t& b) {
                            return a.get_id() == b.get_id();
                        }
                    );
                    nbrs.erase(last, nbrs.end());
                }
            }
        );
    }

private:
    /** @brief Distance function for computing neighbor distances. */
    const dist_func_t& _dist_func;

};  // class RandomEG

}   // namespace cpu
}   // namespace artea
