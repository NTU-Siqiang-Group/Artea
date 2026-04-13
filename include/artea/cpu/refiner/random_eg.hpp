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
#include <tbb/enumerable_thread_specific.h>

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
    using nbr_t = typename RefinerTraitsT::nbr_t;
    using nbr_arr_t = typename RefinerTraitsT::nbr_arr_t;
    using nbr_comp_t = typename RefinerTraitsT::nbr_comp_t;
    using dist_func_t = typename RefinerTraitsT::dist_func_t;
    using random_seq_t = typename RefinerTraitsT::random_seq_t;

    static constexpr nbr_comp_t nbr_comp {};

public:
    /**
     * @brief Construct a new RandomEG object.
     * @param dist_func Distance function for computing neighbor distances.
     */
    RandomEG(const dist_func_t& dist_func)
        : _dist_func(dist_func) {}

    /**
     * @brief Generate random edges for a descent graph.
     * @param refining_graph The graph to initialize with random edges.
     * @param init_nbr_size Number of random neighbors to generate for each vertex.
     */
    template <typename RefiningGraphT>
    auto generate(
        RefiningGraphT& refining_graph,
        const vertex_num_t init_nbr_size
    ) -> void {
        const vertex_num_t num_vertices = refining_graph.get_num_vertices();
        const vector_array_t& vecs_data = refining_graph.get_vecs_data();

        // RandomSeq uses thread-local storage internally, so it's safe to share across threads
        random_seq_t random_seq;

        // Per-thread reusable buffer for random local indices. Indices are
        // drawn in [0, num_vertices) (i.e. local row space when sparse,
        // identical to global vid space when identity-mapped) and then
        // translated to global vids via refining_graph.vid_at(...).
        tbb::enumerable_thread_specific<std::vector<vertex_id_t>>
            tls_random_local_ids([init_nbr_size]{
                return std::vector<vertex_id_t>(init_nbr_size);
            });

        refining_graph.parallel_for_each_vertex(
            [&](const vertex_id_t pivot_vid) {
                auto& random_local_ids = tls_random_local_ids.local();
                nbr_arr_t& nbrs = refining_graph.fetch_nbrs(pivot_vid);
                const vec_ele_t* query_vec = vecs_data.get(pivot_vid);

                random_seq.generate(random_local_ids, num_vertices, init_nbr_size);

                nbrs.clear();

                for (vertex_num_t i = 0; i < init_nbr_size; ++i) {
                    const vertex_id_t nbr_vid =
                        refining_graph.vid_at(random_local_ids[i]);

                    if (nbr_vid == pivot_vid) {
                        continue;
                    }

                    const vec_ele_t* nbr_vec = vecs_data.get(nbr_vid);
                    const distance_t dist = _dist_func(query_vec, nbr_vec);

                    nbrs.emplace_back(nbr_vid, dist, /* is_new = */ true);
                }

                std::sort(nbrs.begin(), nbrs.end(), nbr_comp);

                auto last = std::unique(
                    nbrs.begin(),
                    nbrs.end(),
                    [](const nbr_t& a, const nbr_t& b) {
                        return a.get_vid() == b.get_vid();
                    }
                );
                nbrs.erase(last, nbrs.end());
            }
        );
    }

private:
    /** @brief Distance function for computing neighbor distances. */
    const dist_func_t& _dist_func;

};  // class RandomEG

}   // namespace cpu
}   // namespace artea
