/*
 * @FilePath: /Artea/include/artea/cpu/utils/simd_distance.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2026-05-24 00:00:00
 * @Date: 2025-10-18 19:10:57
 * @Description: SIMD-accelerated distance computation utilities. VecDim
 *               is a required template parameter — NUM_SIMD_CHUNKS is
 *               constexpr so loops fully unroll at -O2. Runtime-dim
 *               dispatch is handled by SIMDDistanceDispatcher (see
 *               simd_distance_dispatcher.hpp).
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <immintrin.h>
#include <artea/common/logger.hpp>

namespace artea {
namespace cpu {

template <typename ComputerTraitsT, std::size_t VecDim, std::size_t UnrollSize = 1>
class SIMDDistance {

    using vec_ele_t = typename ComputerTraitsT::vec_ele_t;
    using distance_t = typename ComputerTraitsT::distance_t;
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

    static_assert(VecDim > 0,
                  "SIMDDistance requires a non-zero VecDim template argument");
    static_assert(VecDim % SIMD_CHUNK_SIZE == 0,
                  "VecDim must be a multiple of SIMD_CHUNK_SIZE (16); pad vectors first");

    /** @brief Number of SIMD chunks processed per distance call (constexpr). */
    static constexpr std::size_t NUM_SIMD_CHUNKS = VecDim / SIMD_CHUNK_SIZE;

public:

    SIMDDistance() = default;

    /**
     * @brief Symmetric distance dispatch by the metric encoded in
     *        ComputerTraits (EUCLIDEAN / DOT / COSINE). Arg order is
     *        irrelevant for all three.
     */
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

    /**
     * @brief Asymmetric FastL2 proxy for ||p - q||^2. Computes
     *        -2*<p, q> + @p p_norm. The omitted ||q||^2 term is a
     *        constant per query, so this preserves argmin / top-K
     *        ranking within a single query but is NOT the true L2^2.
     *
     *        Always available regardless of the metric template
     *        parameter — even an EUCLIDEAN-instantiated functor can
     *        offer this entry to compact-mode search-time callers.
     *        The caller must supply @p p_norm = ||p||^2 (precomputed
     *        via VectorDataset::enable_fast_L2 + get_base_norms()).
     *
     *        Mirrors _impl_euclidean's unroll dispatch on @c unroll_size.
     */
    __attribute__((always_inline))
    auto fast_euclidean(const vec_ele_t* p_vec,
                        const vec_ele_t* q_vec,
                        distance_t       p_norm) const -> distance_t
    {
        const distance_t dot_product = _impl_dot_product(p_vec, q_vec);
        return -2.0f * dot_product + p_norm;
    }

private:

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

        return _mm512_reduce_add_ps(sum_chunk);
    }

    /**
     * @brief AVX-512 dot product with unroll dispatch identical in shape
     *        to @c _impl_euclidean — main difference is no per-chunk
     *        subtraction (FastL2 needs raw <p, q>, not <p-q, p-q>).
     *        Returns the scalar dot product; the caller (fast_euclidean)
     *        applies the @c -2 * dot + p_norm scaling.
     */
    __attribute__((always_inline))
    auto _impl_dot_product(const vec_ele_t* p_vec, const vec_ele_t* q_vec) const -> distance_t {
        __m512 p_chunk, q_chunk, sum_chunk = _mm512_setzero_ps();

        std::size_t i = 0;

        if constexpr (unroll_size == 1) {
            for (; i < NUM_SIMD_CHUNKS; ++i) {
                p_chunk   = _mm512_loadu_ps(p_vec + i * SIMD_CHUNK_SIZE);
                q_chunk   = _mm512_loadu_ps(q_vec + i * SIMD_CHUNK_SIZE);
                sum_chunk = _mm512_fmadd_ps(p_chunk, q_chunk, sum_chunk);
            }
        }
        else if constexpr (unroll_size == 2) {
            __m512 sum_chunk_1 = _mm512_setzero_ps();

            for (; i + 1 < NUM_SIMD_CHUNKS; i += 2) {
                // Chunk 0
                p_chunk   = _mm512_loadu_ps(p_vec + i * SIMD_CHUNK_SIZE);
                q_chunk   = _mm512_loadu_ps(q_vec + i * SIMD_CHUNK_SIZE);
                sum_chunk = _mm512_fmadd_ps(p_chunk, q_chunk, sum_chunk);

                // Chunk 1
                p_chunk     = _mm512_loadu_ps(p_vec + (i + 1) * SIMD_CHUNK_SIZE);
                q_chunk     = _mm512_loadu_ps(q_vec + (i + 1) * SIMD_CHUNK_SIZE);
                sum_chunk_1 = _mm512_fmadd_ps(p_chunk, q_chunk, sum_chunk_1);
            }

            sum_chunk = _mm512_add_ps(sum_chunk, sum_chunk_1);

            for (; i < NUM_SIMD_CHUNKS; ++i) {
                p_chunk   = _mm512_loadu_ps(p_vec + i * SIMD_CHUNK_SIZE);
                q_chunk   = _mm512_loadu_ps(q_vec + i * SIMD_CHUNK_SIZE);
                sum_chunk = _mm512_fmadd_ps(p_chunk, q_chunk, sum_chunk);
            }
        }
        else if constexpr (unroll_size == 4) {
            __m512 sum_chunk_1 = _mm512_setzero_ps();
            __m512 sum_chunk_2 = _mm512_setzero_ps();
            __m512 sum_chunk_3 = _mm512_setzero_ps();

            for (; i + 3 < NUM_SIMD_CHUNKS; i += 4) {
                // Chunk 0
                p_chunk   = _mm512_loadu_ps(p_vec + i * SIMD_CHUNK_SIZE);
                q_chunk   = _mm512_loadu_ps(q_vec + i * SIMD_CHUNK_SIZE);
                sum_chunk = _mm512_fmadd_ps(p_chunk, q_chunk, sum_chunk);

                // Chunk 1
                p_chunk     = _mm512_loadu_ps(p_vec + (i + 1) * SIMD_CHUNK_SIZE);
                q_chunk     = _mm512_loadu_ps(q_vec + (i + 1) * SIMD_CHUNK_SIZE);
                sum_chunk_1 = _mm512_fmadd_ps(p_chunk, q_chunk, sum_chunk_1);

                // Chunk 2
                p_chunk     = _mm512_loadu_ps(p_vec + (i + 2) * SIMD_CHUNK_SIZE);
                q_chunk     = _mm512_loadu_ps(q_vec + (i + 2) * SIMD_CHUNK_SIZE);
                sum_chunk_2 = _mm512_fmadd_ps(p_chunk, q_chunk, sum_chunk_2);

                // Chunk 3
                p_chunk     = _mm512_loadu_ps(p_vec + (i + 3) * SIMD_CHUNK_SIZE);
                q_chunk     = _mm512_loadu_ps(q_vec + (i + 3) * SIMD_CHUNK_SIZE);
                sum_chunk_3 = _mm512_fmadd_ps(p_chunk, q_chunk, sum_chunk_3);
            }

            __m512 sum_01 = _mm512_add_ps(sum_chunk, sum_chunk_1);
            __m512 sum_23 = _mm512_add_ps(sum_chunk_2, sum_chunk_3);
            sum_chunk = _mm512_add_ps(sum_01, sum_23);

            for (; i < NUM_SIMD_CHUNKS; ++i) {
                p_chunk   = _mm512_loadu_ps(p_vec + i * SIMD_CHUNK_SIZE);
                q_chunk   = _mm512_loadu_ps(q_vec + i * SIMD_CHUNK_SIZE);
                sum_chunk = _mm512_fmadd_ps(p_chunk, q_chunk, sum_chunk);
            }
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
