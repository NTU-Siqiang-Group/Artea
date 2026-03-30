/*
 * @FilePath: /Artea/include/artea/cpu/utils/simd_distance.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2026-01-23 18:59:02
 * @Date: 2025-10-18 19:10:57
 * @Description: SIMD-accelerated distance computation utilities.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <immintrin.h>
#include <cassert>
#include <artea/common/logger.hpp>

namespace artea {
namespace cpu {

template <typename ComputerTraitsT, std::size_t UnrollSize = 1>
class SIMDDistance {

    using vec_ele_t = typename ComputerTraitsT::vec_ele_t;
    using distance_t = typename ComputerTraitsT::distance_t;
    using vec_dim_t = typename ComputerTraitsT::vec_dim_t;
    using distance_metrics_t = typename ComputerTraitsT::distance_metrics_t;
    static constexpr distance_metrics_t distance_metrics = ComputerTraitsT::distance_metrics;
    static constexpr std::size_t unroll_size = UnrollSize;

    static constexpr std::size_t SIMD_REGISTER_BITS = 512;
    static constexpr std::size_t SIMD_REGISTER_BYTES = SIMD_REGISTER_BITS / 8;  // 64
    // Number of elements that can be processed in a single SIMD register
    // e.g. we can process 16 elements per chunk for float32 type
    static constexpr std::size_t SIMD_CHUNK_SIZE = [] {
        static_assert(std::is_same_v<vec_ele_t, float>, "SIMDDistance currently supports float type only");
        static_assert(
            SIMD_REGISTER_BYTES % sizeof(vec_ele_t) == 0,
            "SIMD register size must be a multiple of the element size for this utility."
        );
        return SIMD_REGISTER_BYTES / sizeof(vec_ele_t);
    }();    // 16 for float32 type

public:

    SIMDDistance(const vec_dim_t vec_dim) :
        _vec_dim(vec_dim),
        NUM_SIMD_CHUNKS(_vec_dim / SIMD_CHUNK_SIZE),
        NUM_REMAINING_ELES(_vec_dim % SIMD_CHUNK_SIZE),
        _tail_mask(NUM_REMAINING_ELES > 0
            ? static_cast<__mmask16>((1u << NUM_REMAINING_ELES) - 1)
            : static_cast<__mmask16>(0))
    {}

    __attribute__((always_inline))
    auto operator()(const vec_ele_t* vec1, const vec_ele_t* vec2) const -> distance_t {
        if constexpr (distance_metrics == distance_metrics_t::EUCLIDEAN) {
            return _impl_euclidean(vec1, vec2);
        } else if constexpr (distance_metrics == distance_metrics_t::DOT) {
            return _impl_dot(vec1, vec2);
        } else if constexpr (distance_metrics == distance_metrics_t::COSINE) {
            return _impl_cosine(vec1, vec2);
        } else {
            ARTEA_ERROR("Invalid DistanceMetrics");
        }
    }

private:

    const vec_dim_t _vec_dim;
    /** @brief Number of SIMD chunks that can be processed in parallel. */
    const std::size_t NUM_SIMD_CHUNKS;
    /** @brief Number of remaining elements that cannot fill a full SIMD register. */
    const std::size_t NUM_REMAINING_ELES;
    /** @brief AVX-512 mask for the tail (remaining) elements. */
    const __mmask16 _tail_mask;

    __attribute__((always_inline))
    auto _impl_euclidean(const vec_ele_t* vec1, const vec_ele_t* vec2) const -> distance_t {
        // AVX512 implementation
        // sum_chunk serves as the first accumulator (sum_chunk_0)
        __m512 vec1_chunk, vec2_chunk, diff_chunk, sum_chunk = _mm512_set1_ps(0.0f);

        // Process SIMD chunks
        std::size_t i = 0;

        if constexpr (unroll_size == 1) {
            for (; i < NUM_SIMD_CHUNKS; ++i) {
                vec1_chunk = _mm512_loadu_ps(vec1 + i * SIMD_CHUNK_SIZE);
                vec2_chunk = _mm512_loadu_ps(vec2 + i * SIMD_CHUNK_SIZE);
                diff_chunk = _mm512_sub_ps(vec1_chunk, vec2_chunk);
                sum_chunk = _mm512_fmadd_ps(diff_chunk, diff_chunk, sum_chunk);
            }
        }
        else if constexpr (unroll_size == 2) {
            // Define a second accumulator to break dependency chain
            __m512 sum_chunk_1 = _mm512_setzero_ps();

            // Main unrolled loop
            // Ensure we have at least 2 chunks left to process
            for (; i + 1 < NUM_SIMD_CHUNKS; i += 2) {
                // Chunk 0
                vec1_chunk = _mm512_loadu_ps(vec1 + i * SIMD_CHUNK_SIZE);
                vec2_chunk = _mm512_loadu_ps(vec2 + i * SIMD_CHUNK_SIZE);
                diff_chunk = _mm512_sub_ps(vec1_chunk, vec2_chunk);
                sum_chunk = _mm512_fmadd_ps(diff_chunk, diff_chunk, sum_chunk);

                // Chunk 1
                vec1_chunk = _mm512_loadu_ps(vec1 + (i + 1) * SIMD_CHUNK_SIZE);
                vec2_chunk = _mm512_loadu_ps(vec2 + (i + 1) * SIMD_CHUNK_SIZE);
                diff_chunk = _mm512_sub_ps(vec1_chunk, vec2_chunk);
                sum_chunk_1 = _mm512_fmadd_ps(diff_chunk, diff_chunk, sum_chunk_1);
            }

            // Merge accumulators
            sum_chunk = _mm512_add_ps(sum_chunk, sum_chunk_1);

            // Process remaining SIMD chunks (Tail handling for unrolling)
            // This loop handles the cases where NUM_SIMD_CHUNKS is not a multiple of unroll_size
            for (; i < NUM_SIMD_CHUNKS; ++i) {
                vec1_chunk = _mm512_loadu_ps(vec1 + i * SIMD_CHUNK_SIZE);
                vec2_chunk = _mm512_loadu_ps(vec2 + i * SIMD_CHUNK_SIZE);
                diff_chunk = _mm512_sub_ps(vec1_chunk, vec2_chunk);
                sum_chunk = _mm512_fmadd_ps(diff_chunk, diff_chunk, sum_chunk);
            }
        }
        else if constexpr (unroll_size == 4) {
            // Define extra accumulators
            __m512 sum_chunk_1 = _mm512_setzero_ps();
            __m512 sum_chunk_2 = _mm512_setzero_ps();
            __m512 sum_chunk_3 = _mm512_setzero_ps();

            // Main unrolled loop
            // Ensure we have at least 4 chunks left to process
            for (; i + 3 < NUM_SIMD_CHUNKS; i += 4) {
                // Chunk 0
                vec1_chunk = _mm512_loadu_ps(vec1 + i * SIMD_CHUNK_SIZE);
                vec2_chunk = _mm512_loadu_ps(vec2 + i * SIMD_CHUNK_SIZE);
                diff_chunk = _mm512_sub_ps(vec1_chunk, vec2_chunk);
                sum_chunk = _mm512_fmadd_ps(diff_chunk, diff_chunk, sum_chunk);

                // Chunk 1
                vec1_chunk = _mm512_loadu_ps(vec1 + (i + 1) * SIMD_CHUNK_SIZE);
                vec2_chunk = _mm512_loadu_ps(vec2 + (i + 1) * SIMD_CHUNK_SIZE);
                diff_chunk = _mm512_sub_ps(vec1_chunk, vec2_chunk);
                sum_chunk_1 = _mm512_fmadd_ps(diff_chunk, diff_chunk, sum_chunk_1);

                // Chunk 2
                vec1_chunk = _mm512_loadu_ps(vec1 + (i + 2) * SIMD_CHUNK_SIZE);
                vec2_chunk = _mm512_loadu_ps(vec2 + (i + 2) * SIMD_CHUNK_SIZE);
                diff_chunk = _mm512_sub_ps(vec1_chunk, vec2_chunk);
                sum_chunk_2 = _mm512_fmadd_ps(diff_chunk, diff_chunk, sum_chunk_2);

                // Chunk 3
                vec1_chunk = _mm512_loadu_ps(vec1 + (i + 3) * SIMD_CHUNK_SIZE);
                vec2_chunk = _mm512_loadu_ps(vec2 + (i + 3) * SIMD_CHUNK_SIZE);
                diff_chunk = _mm512_sub_ps(vec1_chunk, vec2_chunk);
                sum_chunk_3 = _mm512_fmadd_ps(diff_chunk, diff_chunk, sum_chunk_3);
            }

            // Merge accumulators
            // (sum0 + sum1) + (sum2 + sum3)
            __m512 sum_01 = _mm512_add_ps(sum_chunk, sum_chunk_1);
            __m512 sum_23 = _mm512_add_ps(sum_chunk_2, sum_chunk_3);
            sum_chunk = _mm512_add_ps(sum_01, sum_23);

            // Process remaining SIMD chunks (Tail handling for unrolling)
            // This loop handles the cases where NUM_SIMD_CHUNKS is not a multiple of unroll_size
            for (; i < NUM_SIMD_CHUNKS; ++i) {
                vec1_chunk = _mm512_loadu_ps(vec1 + i * SIMD_CHUNK_SIZE);
                vec2_chunk = _mm512_loadu_ps(vec2 + i * SIMD_CHUNK_SIZE);
                diff_chunk = _mm512_sub_ps(vec1_chunk, vec2_chunk);
                sum_chunk = _mm512_fmadd_ps(diff_chunk, diff_chunk, sum_chunk);
            }
        }

        // Process remaining elements using masked load
        if (NUM_REMAINING_ELES > 0) {
            const std::size_t tail_offset = NUM_SIMD_CHUNKS * SIMD_CHUNK_SIZE;
            __m512 v1_tail = _mm512_maskz_loadu_ps(_tail_mask, vec1 + tail_offset);
            __m512 v2_tail = _mm512_maskz_loadu_ps(_tail_mask, vec2 + tail_offset);
            __m512 diff_tail = _mm512_sub_ps(v1_tail, v2_tail);
            sum_chunk = _mm512_fmadd_ps(diff_tail, diff_tail, sum_chunk);
        }

        return _mm512_reduce_add_ps(sum_chunk);
    }

    __attribute__((always_inline))
    auto _impl_dot(const vec_ele_t* vec1, const vec_ele_t* vec2) const -> distance_t {
        ARTEA_ERROR("Currently DOT distance is not supported");
    }


    __attribute__((always_inline))
    auto _impl_cosine(const vec_ele_t* vec1, const vec_ele_t* vec2) const -> distance_t {
        ARTEA_ERROR("Currently COSINE distance is not supported");
    }

};  // class SIMDDistance

}  // namespace cpu
}   // namespace artea