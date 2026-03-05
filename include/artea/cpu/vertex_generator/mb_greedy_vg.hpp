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

#pragma once

#include <algorithm>
#include <limits>
#include <vector>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

namespace artea {
namespace cpu {

template <typename VertexGeneratorTraitsT>
class MBGreedyVG :
    public VertexGeneratorTraitsT::template vertex_generator_t<MBGreedyVG<VertexGeneratorTraitsT>> {

    using vec_num_t = typename VertexGeneratorTraitsT::vec_num_t;
    using vec_id_t = typename VertexGeneratorTraitsT::vec_id_t;
    using vec_ele_t = typename VertexGeneratorTraitsT::vec_ele_t;
    using vertex_num_t = typename VertexGeneratorTraitsT::vertex_num_t;
    using distance_t = typename VertexGeneratorTraitsT::distance_t;
    using vec_dim_t = typename VertexGeneratorTraitsT::vec_dim_t;
    using vector_array_t = typename VertexGeneratorTraitsT::vector_array_t;
    using dist_func_t = typename VertexGeneratorTraitsT::dist_func_t;
    using approx_rnet_t = typename VertexGeneratorTraitsT::approx_rnet_t;

public:
    MBGreedyVG(const dist_func_t& dist_func) : _dist_func(dist_func) {}

    /**
     * @brief Generate vertex IDs only (without vector data)
     * @return Vector of vertex IDs
     */
    auto gen_id_array(
        const vector_array_t& base_vecs,
        const distance_t min_radius,
        const vertex_num_t max_result_size,
        const vertex_num_t batch_size = 64
    ) -> std::vector<vec_id_t> {
        const vec_num_t total_base_vecs = base_vecs.get_num_vecs();
        std::vector<vec_id_t> result_ids;

        if (total_base_vecs == 0) { return result_ids; }

        result_ids.reserve(max_result_size);
        result_ids.push_back(0);

        // Temporary vector array for distance computation
        vector_array_t temp_vectors(base_vecs.get_vec_dim());
        temp_vectors.reserve(max_result_size);
        temp_vectors.append_vec(base_vecs.get(0));

        vec_num_t batch_start = 0;

        while (result_ids.size() < max_result_size) {
            batch_start += batch_size;
            if (batch_start >= total_base_vecs) { break; }

            const vec_num_t batch_end = std::min(batch_start + batch_size, total_base_vecs);
            const vec_num_t current_batch_size = batch_end - batch_start;

            std::vector<distance_t> candidate_min_distances(current_batch_size);

            tbb::parallel_for(
                tbb::blocked_range<vec_num_t>(0, current_batch_size),
                [&](const tbb::blocked_range<vec_num_t>& r) {
                    for (vec_num_t local_idx = r.begin(); local_idx != r.end(); ++local_idx) {
                        const vec_num_t candidate_idx = batch_start + local_idx;
                        const vec_ele_t* candidate_vec = base_vecs.get(candidate_idx);

                        distance_t min_distance = std::numeric_limits<distance_t>::max();
                        for (vec_num_t ret_idx = 0; ret_idx < temp_vectors.get_num_vecs(); ++ret_idx) {
                            min_distance = std::min(min_distance, _dist_func(candidate_vec, temp_vectors.get(ret_idx)));
                        }

                        candidate_min_distances[local_idx] = min_distance;
                    }
                }
            );

            const auto max_it = std::max_element(candidate_min_distances.begin(), candidate_min_distances.end());
            const distance_t max_distance = *max_it;

            if (max_distance < min_radius) {
                break;
            }

            const vec_num_t best_local_idx = std::distance(candidate_min_distances.begin(), max_it);
            const vec_id_t best_vec_id = batch_start + best_local_idx;
            result_ids.push_back(best_vec_id);
            temp_vectors.append_vec(base_vecs.get(best_vec_id));
        }

        return result_ids;
    }

    /**
     * @brief Default generate method (returns ID array only)
     */
    auto generate(
        const vector_array_t& base_vecs,
        const distance_t min_radius,
        const vertex_num_t max_result_size,
        const vertex_num_t batch_size = 64
    ) -> std::vector<vec_id_t> {
        return gen_id_array(base_vecs, min_radius, max_result_size, batch_size);
    }

private:
    const dist_func_t& _dist_func;

};

}   // namespace cpu
}   // namespace artea
