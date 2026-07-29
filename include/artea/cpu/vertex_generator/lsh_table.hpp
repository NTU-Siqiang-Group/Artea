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
 * @FilePath: /Artea/include/artea/cpu/vertex_generator/lsh_func.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: P-Stable Locality-Sensitive Hashing (LSH) function implementation for vertex generation.
 */

#pragma once

#include <cmath>
#include <utility>
#include <random>
#include <algorithm>

namespace artea {
namespace cpu {

template <typename VertexGeneratorTraitsT>
class LSHTable {
    using vec_id_t = typename VertexGeneratorTraitsT::vec_id_t;
    using vec_num_t = typename VertexGeneratorTraitsT::vec_num_t;
    using vec_ele_t = typename VertexGeneratorTraitsT::vec_ele_t;
    using vec_dim_t = typename VertexGeneratorTraitsT::vec_dim_t;
    using hash_num_t = typename VertexGeneratorTraitsT::hash_num_t;
    using vector_t = typename VertexGeneratorTraitsT::vector_t;
    using vector_array_t = typename VertexGeneratorTraitsT::vector_array_t;
    using dist_func_t = typename VertexGeneratorTraitsT::dist_func_t;

public:
    /** @brief Construct an LSHTable with given projection vector array and offset vector
     *  @note Both projection_vecs and offset_vec should have size equal to num_hashes.
     */
    LSHTable(
        hash_num_t num_hashes,
        vec_ele_t bucket_scale,
        vector_array_t projection_vecs,
        vector_t offset_vec
    ) :
        _num_hashes(num_hashes),
        _bucket_scale(bucket_scale),
        _inv_bucket_scale(static_cast<vec_ele_t>(1.0) / bucket_scale),
        _projection_vecs(std::move(projection_vecs)),
        _offset_vec(std::move(offset_vec)) {}

    LSHTable(LSHTable&&) noexcept = default;
    LSHTable& operator=(LSHTable&&) noexcept = default;
    LSHTable(const LSHTable&) = delete;
    LSHTable& operator=(const LSHTable&) = delete;

    __attribute__((always_inline))
    auto get_projection_vecs() const -> const vector_array_t& {
        return _projection_vecs;
    }

    __attribute__((always_inline))
    auto get_offset_vec() const -> const vector_t& {
        return _offset_vec;
    }

    auto compute(const vec_ele_t* input_vec, vec_ele_t* result_vec) const -> void {
        // Iterate over all hash functions stored in the VectorArray
        for (vec_num_t i = 0; i < _num_hashes; ++i) {
            const vec_ele_t* proj_vec = _projection_vecs.get(i);   // (A)
            vec_ele_t offset = _offset_vec[i];  // (b)
            vec_ele_t projection = _dist_func.dot_product(proj_vec, input_vec); // (A*x)
            // Compute `floor((A*x + b) / r)`
            result_vec[i] = std::floor((projection + offset) * _inv_bucket_scale);
        }
    }

    /**
     * @brief Compute the hash values (projection results) for a given input vector.
     *        Computes: result[i] = (projection_vecs[i] . input_vec) + offset_vec[i]
     * @param input_vec Pointer to the raw data of the input vector.
     * @param result_vec Pointer to the raw data of the result vector
     * @param f
     */
    template <bool read_projection_cache, bool write_projection_cache>
    auto compute(const vec_ele_t* input_vec, vec_ele_t* result_vec, vec_ele_t* projection_cache) const -> void {
        static_assert(not (read_projection_cache and write_projection_cache),
            "read_projection_cache and write_projection_cache can not be simultaneously true, "
            "otherwise this operation is meaningless.");

        // Iterate over all hash functions stored in the VectorArray
        for (vec_num_t i = 0; i < _num_hashes; ++i) {
            vec_ele_t projection;  // (A*x)
            if constexpr (read_projection_cache) {
                projection = projection_cache[i];
            }
            else {
                const vec_ele_t* proj_vec = _projection_vecs.get(i);
                projection = _dist_func.dot_product(proj_vec, input_vec);
            }
            if constexpr (write_projection_cache) {
                projection_cache[i] = projection;
            }
            vec_ele_t offset = _offset_vec[i];  // (b)
            // Compute `floor((A*x + b) / r)`
            result_vec[i] = std::floor((projection + offset) * _inv_bucket_scale);
        }
    }

    /**
     * @brief Update the bucket scale and offset vectors without changing projection vectors.
     *        This is efficient for auto-tuning the bucket scale parameter.
     * @param new_bucket_scale The new bucket width 'r'.
     * @param new_offset_vec The new random offsets 'b', uniformly distributed in [0, r].
     */
    auto update_parameters(vec_ele_t new_bucket_scale, vector_t new_offset_vec) -> void {
        _bucket_scale = new_bucket_scale;
        _inv_bucket_scale = static_cast<vec_ele_t>(1.0) / new_bucket_scale;
        _offset_vec = std::move(new_offset_vec);
    }

    /**
     * @brief Update the bucket scale of an existing LSH table.
     *        This method regenerates the random offsets 'b' to match the new scale range [0, r],
     *        but reuses the expensive orthogonal projection vectors 'A'.
     *
     * @param num_hashes Number of hash functions.
     * @param new_bucket_scale The new bucket width 'r'.
     * @param lsh_table The LSH table instance to be updated.
     */
    auto update_bucket_scale(vec_ele_t new_bucket_scale) -> void {
        // Setup random number generator
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> uniform_dist(0.0, static_cast<double>(new_bucket_scale));
        _bucket_scale = new_bucket_scale;
        _inv_bucket_scale = static_cast<vec_ele_t>(1.0) / new_bucket_scale;
        for (std::size_t i = 0; i < _num_hashes; ++i) {
            _offset_vec[i] = static_cast<vec_ele_t>(uniform_dist(gen));
        }
    }

private:

    /** @brief Number of hash functions */
    hash_num_t _num_hashes;

    /** @brief Projection vectors used for p-stable hashing */
    vector_array_t _projection_vecs;

    /** @brief Offset scalar used for p-stable hashing */
    vector_t _offset_vec;

    /** @brief Bucket scale parameter 'r' */
    vec_ele_t _bucket_scale;

    /** @brief Inverse of bucket scale parameter '1/r' (for fast division) */
    vec_ele_t _inv_bucket_scale;

    /** @brief SIMD distance utility; its raw dot-product API computes projections. */
    dist_func_t _dist_func;

};  // class LSHTable

}   // namespace cpu
}   // namespace artea
