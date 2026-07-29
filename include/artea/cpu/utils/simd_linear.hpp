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
 * @Description: SIMD-accelerated Linear Transformation (Dot Product + Bias).
 * Intended for LSH projection steps and general linear layer computations.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <immintrin.h>
#include <stdexcept>
#include <type_traits>
#include <artea/common/logger.hpp>

namespace artea {
namespace cpu {

/**
 * @brief SIMD implementation for the linear transformation: result = (a . x) + b
 * Where 'a' and 'x' are vectors (dot product) and 'b' is a scalar.
 *
 * @tparam ComputerTraitsT Traits class defining vector element types.
 * @tparam UnrollSize Loop unrolling factor (1, 2, or 4).
 */
template <typename ComputerTraitsT, std::size_t UnrollSize = 1>
class SIMDLinear {

    using vec_ele_t = typename ComputerTraitsT::vec_ele_t;
    using vec_dim_t = typename ComputerTraitsT::vec_dim_t;
    // The result of a linear transformation (dot + bias) is a single scalar
    using result_t = vec_ele_t;

    // Compile-time checks for UnrollSize to avoid invalid template instantiation
    static_assert(UnrollSize == 1 || UnrollSize == 2 || UnrollSize == 4,
                  "UnrollSize must be 1, 2, or 4.");

    static constexpr std::size_t unroll_size = UnrollSize;
    static constexpr std::size_t SIMD_REGISTER_BITS = 512;
    static constexpr std::size_t SIMD_REGISTER_BYTES = SIMD_REGISTER_BITS / 8; // 64

    // Number of elements per SIMD register
    // e.g., 16 elements for float32 (512 bits / 32 bits)
    static constexpr std::size_t SIMD_CHUNK_SIZE = [] {
        static_assert(std::is_same_v<vec_ele_t, float>,
                      "SIMDLinear currently supports float type only (AVX-512 F type).");
        static_assert(SIMD_REGISTER_BYTES % sizeof(vec_ele_t) == 0,
                      "SIMD register size must be a multiple of the element size.");
        return SIMD_REGISTER_BYTES / sizeof(vec_ele_t);
    }();

public:

    /**
     * @brief Construct a new SIMDLinear object.
     *
     * The vector dimension is a compile-time property of ComputerTraitsT
     * (VecDim), so SIMDLinear is a stateless functor with a trivial default
     * constructor — it no longer takes a runtime dimension.
     */
    SIMDLinear() = default;

    /**
     * @brief Computes the linear transformation: (vec_a . vec_x) + b
     *
     * @param vec_a Pointer to the weight vector 'a' (e.g., LSH projection vector).
     * @param vec_x Pointer to the input data vector 'x'.
     * @param b The scalar bias 'b'.
     * @return result_t The scalar result of the operation.
     */
    __attribute__((always_inline))
    auto operator()(const vec_ele_t* vec_a, const vec_ele_t* vec_x, const vec_ele_t b) const -> result_t {
        // Calculate the dot product of vectors a and x
        float dot_product = _impl_dot_product(vec_a, vec_x);
        // Add the scalar bias
        return dot_product + b;
    }

private:

    static constexpr vec_dim_t _vec_dim = ComputerTraitsT::vec_dim;
    static_assert(_vec_dim % SIMD_CHUNK_SIZE == 0,
                  "ComputerTraitsT::vec_dim (VecDim) must be a multiple of "
                  "SIMD_CHUNK_SIZE (16); use the SIMD-padded dimension");
    /** @brief Number of SIMD chunks that can be processed in parallel */
    static constexpr std::size_t NUM_SIMD_CHUNKS = _vec_dim / SIMD_CHUNK_SIZE;

    /**
     * @brief Internal implementation of dot product using AVX-512 with FMA and loop unrolling.
     */
    auto _impl_dot_product(const float* vec_a, const float* vec_x) const -> float {
        // Initialize accumulator to zero
        __m512 sum_chunk = _mm512_setzero_ps();

        __m512 vec_a_chunk;
        __m512 vec_x_chunk;

        std::size_t i = 0;

        // --- Unroll Size = 1 ---
        if constexpr (unroll_size == 1) {
            for (; i < NUM_SIMD_CHUNKS; ++i) {
                vec_a_chunk = _mm512_loadu_ps(vec_a + i * SIMD_CHUNK_SIZE);
                vec_x_chunk = _mm512_loadu_ps(vec_x + i * SIMD_CHUNK_SIZE);
                // Fused Multiply-Add: sum = a * x + sum
                sum_chunk = _mm512_fmadd_ps(vec_a_chunk, vec_x_chunk, sum_chunk);
            }
        }
        // --- Unroll Size = 2 ---
        else if constexpr (unroll_size == 2) {
            // Second accumulator to break dependency chain
            __m512 sum_chunk_1 = _mm512_setzero_ps();

            // Main loop processing 2 chunks per iteration
            for (; i + 1 < NUM_SIMD_CHUNKS; i += 2) {
                // Chunk 0
                vec_a_chunk = _mm512_loadu_ps(vec_a + i * SIMD_CHUNK_SIZE);
                vec_x_chunk = _mm512_loadu_ps(vec_x + i * SIMD_CHUNK_SIZE);
                sum_chunk = _mm512_fmadd_ps(vec_a_chunk, vec_x_chunk, sum_chunk);
                // Chunk 1
                vec_a_chunk = _mm512_loadu_ps(vec_a + (i + 1) * SIMD_CHUNK_SIZE);
                vec_x_chunk = _mm512_loadu_ps(vec_x + (i + 1) * SIMD_CHUNK_SIZE);
                sum_chunk_1 = _mm512_fmadd_ps(vec_a_chunk, vec_x_chunk, sum_chunk_1);
            }

            // Merge accumulators
            sum_chunk = _mm512_add_ps(sum_chunk, sum_chunk_1);

            // Handle remaining chunks (if any)
            for (; i < NUM_SIMD_CHUNKS; ++i) {
                vec_a_chunk = _mm512_loadu_ps(vec_a + i * SIMD_CHUNK_SIZE);
                vec_x_chunk = _mm512_loadu_ps(vec_x + i * SIMD_CHUNK_SIZE);
                sum_chunk = _mm512_fmadd_ps(vec_a_chunk, vec_x_chunk, sum_chunk);
            }
        }
        // --- Unroll Size = 4 ---
        else if constexpr (unroll_size == 4) {
            // Extra accumulators
            __m512 sum_chunk_1 = _mm512_setzero_ps();
            __m512 sum_chunk_2 = _mm512_setzero_ps();
            __m512 sum_chunk_3 = _mm512_setzero_ps();

            // Main loop processing 4 chunks per iteration
            for (; i + 3 < NUM_SIMD_CHUNKS; i += 4) {
                // Chunk 0
                vec_a_chunk = _mm512_loadu_ps(vec_a + i * SIMD_CHUNK_SIZE);
                vec_x_chunk = _mm512_loadu_ps(vec_x + i * SIMD_CHUNK_SIZE);
                sum_chunk = _mm512_fmadd_ps(vec_a_chunk, vec_x_chunk, sum_chunk);
                // Chunk 1
                vec_a_chunk = _mm512_loadu_ps(vec_a + (i + 1) * SIMD_CHUNK_SIZE);
                vec_x_chunk = _mm512_loadu_ps(vec_x + (i + 1) * SIMD_CHUNK_SIZE);
                sum_chunk_1 = _mm512_fmadd_ps(vec_a_chunk, vec_x_chunk, sum_chunk_1);
                // Chunk 2
                vec_a_chunk = _mm512_loadu_ps(vec_a + (i + 2) * SIMD_CHUNK_SIZE);
                vec_x_chunk = _mm512_loadu_ps(vec_x + (i + 2) * SIMD_CHUNK_SIZE);
                sum_chunk_2 = _mm512_fmadd_ps(vec_a_chunk, vec_x_chunk, sum_chunk_2);
                // Chunk 3
                vec_a_chunk = _mm512_loadu_ps(vec_a + (i + 3) * SIMD_CHUNK_SIZE);
                vec_x_chunk = _mm512_loadu_ps(vec_x + (i + 3) * SIMD_CHUNK_SIZE);
                sum_chunk_3 = _mm512_fmadd_ps(vec_a_chunk, vec_x_chunk, sum_chunk_3);
            }

            // Merge accumulators: tree reduction ((0+1) + (2+3))
            __m512 sum_01 = _mm512_add_ps(sum_chunk, sum_chunk_1);
            __m512 sum_23 = _mm512_add_ps(sum_chunk_2, sum_chunk_3);
            sum_chunk = _mm512_add_ps(sum_01, sum_23);

            // Handle remaining chunks (if any)
            for (; i < NUM_SIMD_CHUNKS; ++i) {
                vec_a_chunk = _mm512_loadu_ps(vec_a + i * SIMD_CHUNK_SIZE);
                vec_x_chunk = _mm512_loadu_ps(vec_x + i * SIMD_CHUNK_SIZE);
                sum_chunk = _mm512_fmadd_ps(vec_a_chunk, vec_x_chunk, sum_chunk);
            }
        }

        // Horizontal reduction to sum all elements in the register
        return _mm512_reduce_add_ps(sum_chunk);
    }

}; // class SIMDLinear

} // namespace cpu
} // namespace artea