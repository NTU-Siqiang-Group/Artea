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

#include <vector>
#include <algorithm>
#include <artea/cpu/utils/random_seq_nr.hpp>

namespace artea {
namespace cpu {

template <typename VertexGeneratorTraitsT>
class RandomVG :
    public VertexGeneratorTraitsT::template vertex_generator_t<RandomVG<VertexGeneratorTraitsT>> {

    using vec_num_t = typename VertexGeneratorTraitsT::vec_num_t;
    using vec_id_t = typename VertexGeneratorTraitsT::vec_id_t;
    using vec_ele_t = typename VertexGeneratorTraitsT::vec_ele_t;
    using vertex_num_t = typename VertexGeneratorTraitsT::vertex_num_t;
    using vec_dim_t = typename VertexGeneratorTraitsT::vec_dim_t;
    using vector_array_t = typename VertexGeneratorTraitsT::vector_array_t;
    using vertex_subset_t = typename VertexGeneratorTraitsT::vertex_subset_t;
    using random_seq_nr_t = typename VertexGeneratorTraitsT::random_seq_nr_t;

public:
    RandomVG() = default;

    /**
     * @brief Generate random vertices with vector data
     * @param vecs_data The vector array to generate vertices from
     * @param result_size Number of vertices to generate
     * @return VertexSubset containing random vertices
     */
    auto generate_impl(
        const vector_array_t& vecs_data,
        const vertex_num_t result_size
    ) -> vertex_subset_t {
        const vec_num_t total_vecs = vecs_data.get_num_vecs();

        vertex_subset_t vertex_subset(vecs_data.get_vec_dim());

        if (total_vecs == 0 || result_size == 0) {
            return vertex_subset;
        }

        // Determine actual result size (cannot exceed total vecs)
        const vertex_num_t actual_result_size = std::min(result_size, static_cast<vertex_num_t>(total_vecs));

        // Generate random indices using RandomSeqNR
        random_seq_nr_t random_seq_nr(total_vecs);
        auto random_ids = random_seq_nr.generate(actual_result_size);

        // Copy to vec_ids and sort for better cache locality
        vertex_subset.vec_ids.insert(vertex_subset.vec_ids.end(), random_ids.begin(), random_ids.end());
        std::sort(vertex_subset.vec_ids.begin(), vertex_subset.vec_ids.end());

        // Extract vectors using extract_subset
        vertex_subset.vecs_data = vecs_data.extract_subset(vertex_subset.vec_ids);

        return vertex_subset;
    }

private:

};

}   // namespace cpu
}   // namespace artea