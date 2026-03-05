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
#include <artea/cpu/utils/random_seq.hpp>

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
    using random_seq_t = RandomSeq<VertexGeneratorTraitsT>;

public:
    RandomVG() = default;

    /**
     * @brief Generate random vertex IDs only (without vector data)
     * @return Vector of random vertex IDs
     */
    auto gen_id_array(
        const vector_array_t& base_vecs,
        const vertex_num_t result_size
    ) -> std::vector<vec_id_t> {
        const vec_num_t total_base_vecs = base_vecs.get_num_vecs();

        if (total_base_vecs == 0 || result_size == 0) {
            return std::vector<vec_id_t>();
        }

        // Determine actual result size (cannot exceed total base vecs)
        const vertex_num_t actual_result_size = std::min(result_size, static_cast<vertex_num_t>(total_base_vecs));

        // Generate random indices using RandomSeq
        random_seq_t random_seq(total_base_vecs);
        std::vector<vec_id_t> random_ids(actual_result_size);
        random_seq.generate(random_ids, actual_result_size);

        return random_ids;
    }

    /**
     * @brief Default generate method (returns ID array only)
     */
    auto generate(
        const vector_array_t& base_vecs,
        const vertex_num_t result_size
    ) -> std::vector<vec_id_t> {
        return gen_id_array(base_vecs, result_size);
    }

private:

};

}   // namespace cpu
}   // namespace artea