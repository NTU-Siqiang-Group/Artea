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
 * @FilePath: /Artea/include/artea/cpu/vertex_generator/ortho_lsh_generator.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description:
 */

#pragma once

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <Eigen/Dense>

namespace artea {
namespace cpu {

template <typename VertexGeneratorTraitsT>
class OrthoLSHGenerator {

    using vec_id_t = typename VertexGeneratorTraitsT::vec_id_t;
    using vec_num_t = typename VertexGeneratorTraitsT::vec_num_t;
    using vec_ele_t = typename VertexGeneratorTraitsT::vec_ele_t;
    using vec_dim_t = typename VertexGeneratorTraitsT::vec_dim_t;
    using hash_num_t = typename VertexGeneratorTraitsT::hash_num_t;
    using vector_t = typename VertexGeneratorTraitsT::vector_t;
    using vector_array_t = typename VertexGeneratorTraitsT::vector_array_t;
    using lsh_table_t = typename VertexGeneratorTraitsT::lsh_table_t;
    using fma_func_t = typename VertexGeneratorTraitsT::fma_func_t;
    using distance_metrics_t = typename VertexGeneratorTraitsT::distance_metrics_t;
    static constexpr distance_metrics_t distance_metrics = VertexGeneratorTraitsT::distance_metrics;

    static_assert(
        distance_metrics == distance_metrics_t::EUCLIDEAN,
        "OrthoLSHGenerator currently supports EUCLIDEAN distance only."
    );

public:
    OrthoLSHGenerator(const fma_func_t& fma_func) : _fma_func(fma_func) {}

    /**
     * @brief Generate orthogonal LSH projection vectors.
     * @param dim The dimensionality of the vectors.
     * @param num_hashes The number of hash functions (projection vectors) to generate.
     * @return A vector array containing the generated orthogonal p-stable LSH random projection vectors.
     */
    auto generate(vec_dim_t dim, hash_num_t num_hashes, vec_ele_t bucket_scale) -> lsh_table_t {
        vector_array_t projection_vecs(num_hashes, dim);
        vector_t offset_vec(num_hashes);
        std::size_t num_blocks = (num_hashes + dim - 1) / dim;
        /** The scale factor ensures that the LSH buckets are uniformly distributed. */
        const float scale_factor = std::sqrt(static_cast<float>(dim));

        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> norm_dist(0.0, 1.0);
        std::uniform_real_distribution<double> uniform_dist(0.0, static_cast<double>(bucket_scale));
        for (std::size_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
            // Using MatrixXd ensures high precision during decomposition.
            Eigen::MatrixXd M(dim, dim);
            for (int r = 0; r < dim; ++r) {
                for (int c = 0; c < dim; ++c) {
                    M(r, c) = norm_dist(gen);
                }
            }
            // Perform QR decomposition: M = QR
            Eigen::HouseholderQR<Eigen::MatrixXd> qr(M);
            Eigen::MatrixXd Q = qr.householderQ();

            // Copy Q's rows into VectorArray and generate offsets
            std::size_t start_hash_idx = block_idx * dim;
            std::size_t end_hash_idx = std::min(static_cast<std::size_t>(num_hashes), start_hash_idx + dim);
            for (std::size_t i = start_hash_idx; i < end_hash_idx; ++i) {
                // Map global index 'i' to row index in Q
                int row_in_Q = static_cast<int>(i - start_hash_idx);
                vec_ele_t* dest_ptr = projection_vecs.get(i);
                // Copy data from Eigen to raw pointer and scale
                for (vec_dim_t d = 0; d < dim; ++d) {
                    dest_ptr[d] = static_cast<vec_ele_t>(Q(row_in_Q, d) * scale_factor);
                }
                // Generate random offset 'b' in [0, bucket_scale]
                offset_vec[i] = static_cast<vec_ele_t>(uniform_dist(gen));
            }
        }

        return lsh_table_t(
            num_hashes,
            bucket_scale,
            std::move(projection_vecs),
            std::move(offset_vec),
            _fma_func
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

private:
    const fma_func_t& _fma_func;

};  // class OrthoLSHGenerator

}   // namespace cpu
}   // namespace artea