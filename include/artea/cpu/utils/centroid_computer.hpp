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
 * @FilePath: /Artea/include/artea/cpu/utils/centroid_computer.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Utility for computing centroid of a vector array using SIMD vectorization.
 */

#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>

namespace artea {
namespace cpu {

/**
 * @brief Computes the centroid (mean vector) of a vector array using SIMD vectorization.
 * @tparam BaseTraitsT The base traits type.
 */
template <typename BaseTraitsT>
class CentroidComputer {

    using vec_ele_t = typename BaseTraitsT::vec_ele_t;
    using vector_array_t = typename BaseTraitsT::vector_array_t;
    using vertex_num_t = typename BaseTraitsT::vertex_num_t;
    using vec_dim_t = typename BaseTraitsT::vec_dim_t;

public:
    /**
     * @brief Compute the centroid of a vector array using SIMD vectorization.
     * @param vecs The vector array.
     * @return std::vector<vec_ele_t> The centroid vector.
     */
    static auto compute(const vector_array_t& vecs) -> std::vector<vec_ele_t> {
        const vertex_num_t num_vecs = vecs.get_num_vecs();
        const vec_dim_t vec_dim = vecs.get_vec_dim();

        if (num_vecs == 0) {
            return std::vector<vec_ele_t>(vec_dim, 0.0f);
        }

        std::vector<vec_ele_t> centroid(vec_dim, 0.0f);

        // Sum all vectors using SIMD vectorization
        for (vertex_num_t i = 0; i < num_vecs; ++i) {
            const vec_ele_t* vec = vecs.get(i);

            #pragma omp simd
            for (vec_dim_t d = 0; d < vec_dim; ++d) {
                centroid[d] += vec[d];
            }
        }

        // Divide by number of vectors to get mean using SIMD vectorization
        const vec_ele_t inv_num_vecs = static_cast<vec_ele_t>(1.0) / static_cast<vec_ele_t>(num_vecs);

        #pragma omp simd
        for (vec_dim_t d = 0; d < vec_dim; ++d) {
            centroid[d] *= inv_num_vecs;
        }

        return centroid;
    }

};  // class CentroidComputer

}   // namespace cpu
}   // namespace artea
