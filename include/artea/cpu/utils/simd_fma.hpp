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

#include <cstddef>
#include <cstdint>
#include <immintrin.h>
#include <stdexcept>
#include <type_traits>
#include <artea/common/logger.hpp>

namespace artea {
namespace cpu {

template <typename ComputerTraitsT, std::size_t UnrollSize = 1>
class SIMDFMA {

    using vec_ele_t = typename ComputerTraitsT::vec_ele_t;
    using vec_dim_t = typename ComputerTraitsT::vec_dim_t;
    using result_t = vec_ele_t;

    static constexpr std::size_t unroll_size = UnrollSize;

    static constexpr std::size_t SIMD_REGISTER_BITS = 512;
    static constexpr std::size_t SIMD_REGISTER_BYTES = SIMD_REGISTER_BITS / 8;  // 64
    // Number of elements that can be processed in a single SIMD register
    // e.g. we can process 16 elements per chunk for float32 type
    static constexpr std::size_t SIMD_CHUNK_SIZE = [] {
        static_assert(std::is_same_v<vec_ele_t, float>, "SIMDFMA currently supports float type only");
        static_assert(
            SIMD_REGISTER_BYTES % sizeof(vec_ele_t) == 0,
            "SIMD register size must be a multiple of the element size for this utility."
        );
        return SIMD_REGISTER_BYTES / sizeof(vec_ele_t);
    }();    // 16 for float32 type

public:

    SIMDFMA(const vec_dim_t vec_dim) :
        _vec_dim(vec_dim),
        NUM_SIMD_CHUNKS(_vec_dim / SIMD_CHUNK_SIZE),
        NUM_REMAINING_ELES(_vec_dim % SIMD_CHUNK_SIZE)
    {
        ArteaLogger logger("SIMDFMA", LogLevelT::INFO);
        if (vec_dim % SIMD_CHUNK_SIZE != 0) {
            logger.error(
                "Vector dimension must be a multiple of SIMD chunk size (e.g. 16 for float type)"
            );
        }
    }

    __attribute__((always_inline))
    auto operator()(const vec_ele_t* vec1, const vec_ele_t* vec2) const -> result_t {
        return _impl_dot_float(vec1, vec2);
    }

private:

    const vec_dim_t _vec_dim;
    /** @brief Number of SIMD chunks that can be processed in parallel */
    const std::size_t NUM_SIMD_CHUNKS;
    /** @brief Number of remaining elements that cannot be processed in parallel */
    const std::size_t NUM_REMAINING_ELES;

    auto _impl_dot_float(const float* vec1, const float* vec2) const -> float {
        // AVX512 implementation for Fused Multiply-Add (Dot Product)
        // sum_chunk serves as the first accumulator (sum_chunk_0)
        __m512 vec1_chunk, vec2_chunk, sum_chunk = _mm512_setzero_ps();

        // Process SIMD chunks
        std::size_t i = 0;

        if constexpr (unroll_size == 1) {
            for (; i < NUM_SIMD_CHUNKS; ++i) {
                vec1_chunk = _mm512_loadu_ps(vec1 + i * SIMD_CHUNK_SIZE);
                vec2_chunk = _mm512_loadu_ps(vec2 + i * SIMD_CHUNK_SIZE);
                // sum = vec1 * vec2 + sum
                sum_chunk = _mm512_fmadd_ps(vec1_chunk, vec2_chunk, sum_chunk);
            }
        }
        else if constexpr (unroll_size == 2) {
            // Define a second accumulator to break dependency chain
            __m512 sum_chunk_1 = _mm512_setzero_ps();

            // Main unrolled loop
            for (; i + 1 < NUM_SIMD_CHUNKS; i += 2) {
                // Chunk 0
                vec1_chunk = _mm512_loadu_ps(vec1 + i * SIMD_CHUNK_SIZE);
                vec2_chunk = _mm512_loadu_ps(vec2 + i * SIMD_CHUNK_SIZE);
                sum_chunk = _mm512_fmadd_ps(vec1_chunk, vec2_chunk, sum_chunk);
                // Chunk 1
                vec1_chunk = _mm512_loadu_ps(vec1 + (i + 1) * SIMD_CHUNK_SIZE);
                vec2_chunk = _mm512_loadu_ps(vec2 + (i + 1) * SIMD_CHUNK_SIZE);
                sum_chunk_1 = _mm512_fmadd_ps(vec1_chunk, vec2_chunk, sum_chunk_1);
            }

            // Merge accumulators
            sum_chunk = _mm512_add_ps(sum_chunk, sum_chunk_1);

            // Process remaining SIMD chunks (Tail handling for unrolling)
            for (; i < NUM_SIMD_CHUNKS; ++i) {
                vec1_chunk = _mm512_loadu_ps(vec1 + i * SIMD_CHUNK_SIZE);
                vec2_chunk = _mm512_loadu_ps(vec2 + i * SIMD_CHUNK_SIZE);
                sum_chunk = _mm512_fmadd_ps(vec1_chunk, vec2_chunk, sum_chunk);
            }
        }
        else if constexpr (unroll_size == 4) {
            // Define extra accumulators
            __m512 sum_chunk_1 = _mm512_setzero_ps();
            __m512 sum_chunk_2 = _mm512_setzero_ps();
            __m512 sum_chunk_3 = _mm512_setzero_ps();

            // Main unrolled loop
            for (; i + 3 < NUM_SIMD_CHUNKS; i += 4) {
                // Chunk 0
                vec1_chunk = _mm512_loadu_ps(vec1 + i * SIMD_CHUNK_SIZE);
                vec2_chunk = _mm512_loadu_ps(vec2 + i * SIMD_CHUNK_SIZE);
                sum_chunk = _mm512_fmadd_ps(vec1_chunk, vec2_chunk, sum_chunk);
                // Chunk 1
                vec1_chunk = _mm512_loadu_ps(vec1 + (i + 1) * SIMD_CHUNK_SIZE);
                vec2_chunk = _mm512_loadu_ps(vec2 + (i + 1) * SIMD_CHUNK_SIZE);
                sum_chunk_1 = _mm512_fmadd_ps(vec1_chunk, vec2_chunk, sum_chunk_1);
                // Chunk 2
                vec1_chunk = _mm512_loadu_ps(vec1 + (i + 2) * SIMD_CHUNK_SIZE);
                vec2_chunk = _mm512_loadu_ps(vec2 + (i + 2) * SIMD_CHUNK_SIZE);
                sum_chunk_2 = _mm512_fmadd_ps(vec1_chunk, vec2_chunk, sum_chunk_2);
                // Chunk 3
                vec1_chunk = _mm512_loadu_ps(vec1 + (i + 3) * SIMD_CHUNK_SIZE);
                vec2_chunk = _mm512_loadu_ps(vec2 + (i + 3) * SIMD_CHUNK_SIZE);
                sum_chunk_3 = _mm512_fmadd_ps(vec1_chunk, vec2_chunk, sum_chunk_3);
            }

            // Merge accumulators
            // (sum0 + sum1) + (sum2 + sum3)
            __m512 sum_01 = _mm512_add_ps(sum_chunk, sum_chunk_1);
            __m512 sum_23 = _mm512_add_ps(sum_chunk_2, sum_chunk_3);
            sum_chunk = _mm512_add_ps(sum_01, sum_23);

            // Process remaining SIMD chunks (Tail handling for unrolling)
            for (; i < NUM_SIMD_CHUNKS; ++i) {
                vec1_chunk = _mm512_loadu_ps(vec1 + i * SIMD_CHUNK_SIZE);
                vec2_chunk = _mm512_loadu_ps(vec2 + i * SIMD_CHUNK_SIZE);
                sum_chunk = _mm512_fmadd_ps(vec1_chunk, vec2_chunk, sum_chunk);
            }
        }

        // Horizontal reduction
        return _mm512_reduce_add_ps(sum_chunk);
    }

};  // class SIMDFMA

}   // namespace cpu
}   // namespace artea