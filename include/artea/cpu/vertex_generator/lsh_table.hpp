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
    using fma_func_t = typename VertexGeneratorTraitsT::fma_func_t;

public:
    /** @brief Construct an LSHTable with given projection vector array and offset vector
     *  @note Both projection_vecs and offset_vec should have size equal to num_hashes.
     *  @note std::move is compulsory to avoid unnecessary copies:
     *        `LSHTable<Traits> lsh(num_hashes, bucket_scale, std::move(proj), std::move(off), fma_func);`
     */
    LSHTable(
        hash_num_t num_hashes,
        vec_ele_t bucket_scale,
        vector_array_t projection_vecs,
        vector_t offset_vec,
        const fma_func_t& fma_func
    ) :
        _num_hashes(num_hashes),
        _bucket_scale(bucket_scale),
        _inv_bucket_scale(static_cast<vec_ele_t>(1.0) / bucket_scale),
        _projection_vecs(std::move(projection_vecs)),
        _offset_vec(offset_vec),
        _fma_func(fma_func) {}

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
            vec_ele_t fma_val = _fma_func(proj_vec, input_vec); // (A*x)
            // Compute `floor((A*x + b) / r)`
            result_vec[i] = std::floor((fma_val + offset) * _inv_bucket_scale);
        }
    }

    /**
     * @brief Compute the hash values (projection results) for a given input vector.
     *        Computes: result[i] = (projection_vecs[i] . input_vec) + offset_vec[i]
     * @param input_vec Pointer to the raw data of the input vector.
     * @param result_vec Pointer to the raw data of the result vector
     * @param f
     */
    template <bool read_fma_cache, bool write_fma_cache>
    auto compute(const vec_ele_t* input_vec, vec_ele_t* result_vec, vec_ele_t* fma_cache) const -> void {
        static_assert(not (read_fma_cache and write_fma_cache),
            "read_fma_cache and write_fma_cache can not be simultanuously set True, "
            "otherwise this operation is meaningless.");

        // Iterate over all hash functions stored in the VectorArray
        for (vec_num_t i = 0; i < _num_hashes; ++i) {
            vec_ele_t fma_val;  // (A*x)
            if constexpr (read_fma_cache) {
                fma_val = fma_cache[i];
            }
            else {
                const vec_ele_t* proj_vec = _projection_vecs.get(i);
                fma_val = _fma_func(proj_vec, input_vec);
            }
            if constexpr (write_fma_cache) {
                fma_cache[i] = fma_val;
            }
            vec_ele_t offset = _offset_vec[i];  // (b)
            // Compute `floor((A*x + b) / r)`
            result_vec[i] = std::floor((fma_val + offset) * _inv_bucket_scale);
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

private:

    hash_num_t _num_hashes;

    /** @brief Projection vectors used for p-stable hashing */
    vector_array_t _projection_vecs;

    /** @brief Offset scalar used for p-stable hashing */
    vector_t _offset_vec;

    vec_ele_t _bucket_scale;

    vec_ele_t _inv_bucket_scale;

    /** @brief FMA function used for projection */
    fma_func_t _fma_func;

};  // class LSHTable

}   // namespace cpu
}   // namespace artea