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
 * @FilePath: /Artea/include/artea/cpu/vertex_generator/pstable_lsh_generator.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Standard P-Stable LSH generator without orthogonalization.
 */

#pragma once

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <omp.h>

namespace artea {
namespace cpu {

template <typename VertexGeneratorTraitsT>
class PStableLSHGenerator {

    using vec_id_t = typename VertexGeneratorTraitsT::vec_id_t;
    using vec_num_t = typename VertexGeneratorTraitsT::vec_num_t;
    using vec_ele_t = typename VertexGeneratorTraitsT::vec_ele_t;
    using vec_dim_t = typename VertexGeneratorTraitsT::vec_dim_t;
    using hash_num_t = typename VertexGeneratorTraitsT::hash_num_t;
    using vector_t = typename VertexGeneratorTraitsT::vector_t;
    using vector_array_t = typename VertexGeneratorTraitsT::vector_array_t;
    using lsh_table_t = typename VertexGeneratorTraitsT::lsh_table_t;
    using distance_metrics_t = typename VertexGeneratorTraitsT::distance_metrics_t;
    static constexpr distance_metrics_t distance_metrics = VertexGeneratorTraitsT::distance_metrics;

    // Check if the distance metric is compatible with Gaussian distribution
    static_assert(
        distance_metrics == distance_metrics_t::EUCLIDEAN,
        "PStableLSHGenerator currently supports EUCLIDEAN distance only (implies Gaussian distribution)."
    );

public:
    PStableLSHGenerator() = default;

    /**
     * @brief Generate standard p-stable LSH projection vectors.
     *        Unlike OrthoLSHGenerator, this implementation generates independent random vectors
     *        without enforcing orthogonality via QR decomposition.
     *
     * @param dim The dimensionality of the vectors.
     * @param num_hashes The number of hash functions (projection vectors) to generate.
     * @param bucket_scale The width of the projection bucket (parameter 'r').
     * @return A vector array containing the generated p-stable LSH random projection vectors.
     */
    auto generate(vec_dim_t dim, hash_num_t num_hashes, vec_ele_t bucket_scale) -> lsh_table_t {
        // Initialize storage
        vector_array_t projection_vecs(num_hashes, dim);
        vector_t offset_vec(num_hashes);
        vec_ele_t* offset_data = offset_vec.data();

        #pragma omp parallel
        {
            // Thread-local random number generators to ensure thread safety
            std::random_device rd;
            std::mt19937 gen(rd() + omp_get_thread_num());
            // For Euclidean distance (L2 norm), p-stable distribution is Gaussian (Normal)
            std::normal_distribution<vec_ele_t> norm_dist(0.0, 1.0);
            // The offset 'b' is chosen uniformly from [0, r]
            std::uniform_real_distribution<vec_ele_t> uniform_dist(0.0, bucket_scale);

            // Parallel loop: Each hash function is generated independently.
            #pragma omp for schedule(static)
            for (hash_num_t i = 0; i < num_hashes; ++i) {
                // Generate Projection Vector 'A'
                vec_ele_t* vec_ptr = projection_vecs.get(i);
                for (vec_dim_t d = 0; d < dim; ++d) {
                    vec_ptr[d] = norm_dist(gen);
                }
                // Generate Offset 'b'
                offset_data[i] = uniform_dist(gen);
            }
        } // End of parallel region

        return lsh_table_t(
            num_hashes,
            bucket_scale,
            std::move(projection_vecs),
            std::move(offset_vec)
        );
    }

    /**
     * @brief Update the bucket scale of an existing LSH table.
     *        This method regenerates the random offsets 'b' to match the new scale range [0, r],
     *        but reuses the existing projection vectors 'A'.
     *
     * @param dim Dimensionality of vectors (unused here but kept for interface consistency).
     * @param num_hashes Number of hash functions.
     * @param new_bucket_scale The new bucket width 'r'.
     * @param lsh_table The LSH table instance to be updated.
     */
    auto update_bucket_scale(
        hash_num_t num_hashes,
        vec_ele_t new_bucket_scale,
        lsh_table_t& lsh_table
    ) -> void {
        vector_t new_offset_vec(num_hashes);
        vec_ele_t* new_offset_data = new_offset_vec.data();

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<vec_ele_t> uniform_dist(0.0, new_bucket_scale);

        for (hash_num_t i = 0; i < num_hashes; ++i) {
            new_offset_data[i] = uniform_dist(gen);
        }
        // Update the table with new parameters
        lsh_table.update_parameters(new_bucket_scale, std::move(new_offset_vec));
    }

};  // class PStableLSHGenerator

}   // namespace cpu
}   // namespace artea
