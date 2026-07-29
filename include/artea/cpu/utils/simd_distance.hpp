/*
 * @FilePath: /Artea/include/artea/cpu/utils/simd_distance.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2026-03-30 16:41:05
 * @Date: 2025-10-18 19:10:57
 * @Description: SIMD-accelerated distance computation utilities.
 */

#pragma once

#include <cmath>
#include <cstddef>
#include <immintrin.h>
#include <type_traits>
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

    static_assert(
        UnrollSize == 1 || UnrollSize == 2 || UnrollSize == 4,
        "UnrollSize must be 1, 2, or 4."
    );

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

    // Compile-time SIMD-padded dimension, carried by ComputerTraitsT (VecDim).
    // Vectors are padded to a multiple of SIMD_CHUNK_SIZE upstream
    // (VectorDataset::_pad_to_simd_alignment); the masked-tail path is gone, so
    // VecDim MUST already be that padded dimension. Checked at compile time.
    static constexpr vec_dim_t _vec_dim = ComputerTraitsT::vec_dim;
    static_assert(_vec_dim % SIMD_CHUNK_SIZE == 0,
                  "ComputerTraitsT::vec_dim (VecDim) must be a multiple of "
                  "SIMD_CHUNK_SIZE (16); use the SIMD-padded dimension");

public:

    // The vector dimension is a compile-time property of ComputerTraitsT, so
    // SIMDDistance is a stateless functor with a trivial default constructor —
    // it no longer takes a runtime dimension.
    SIMDDistance() = default;

    __attribute__((always_inline))
    auto operator()(const vec_ele_t* vec1, const vec_ele_t* vec2) const -> distance_t {
        if constexpr (distance_metrics == distance_metrics_t::EUCLIDEAN) {
            return _impl_euclidean(vec1, vec2);
        } else if constexpr (distance_metrics == distance_metrics_t::DOT) {
            // Inner-product search is expressed as a distance, so larger dot
            // products become smaller (negated) distances.
            return static_cast<distance_t>(-dot_product(vec1, vec2));
        } else if constexpr (distance_metrics == distance_metrics_t::COSINE) {
            return _impl_cosine(vec1, vec2);
        } else {
            ARTEA_ERROR("Invalid DistanceMetrics");
        }
    }

    /**
     * @brief Compute the raw (non-negated) dot product.
     *
     * This is intentionally distinct from operator() for the DOT metric:
     * operator() returns -dot so the result follows the library's
     * smaller-is-closer distance convention, while projection users such as
     * p-stable LSH need the algebraic dot product itself.
     */
    __attribute__((always_inline))
    auto dot_product(const vec_ele_t* vec1, const vec_ele_t* vec2) const -> distance_t {
        return _impl_dot_product(vec1, vec2);
    }

private:

    /** @brief Number of SIMD chunks that can be processed in parallel
     *         (compile-time, derived from the padded VecDim). */
    static constexpr std::size_t NUM_SIMD_CHUNKS = _vec_dim / SIMD_CHUNK_SIZE;

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

    __attribute__((always_inline))
    auto _impl_dot_product(const vec_ele_t* vec1, const vec_ele_t* vec2) const -> distance_t {
        // Raw dot-product kernel shared by DOT distance and projection users.
        // Vectors are SIMD-padded upstream; the zero tail adds nothing.
        __m512 sum_chunks[unroll_size];

        #pragma GCC unroll 4
        for (std::size_t u = 0; u < unroll_size; ++u) {
            sum_chunks[u] = _mm512_setzero_ps();
        }

        std::size_t i = 0;
        for (; i + unroll_size <= NUM_SIMD_CHUNKS; i += unroll_size) {
            #pragma GCC unroll 4
            for (std::size_t u = 0; u < unroll_size; ++u) {
                const std::size_t chunk_idx = i + u;
                const __m512 vec1_chunk = _mm512_loadu_ps(vec1 + chunk_idx * SIMD_CHUNK_SIZE);
                const __m512 vec2_chunk = _mm512_loadu_ps(vec2 + chunk_idx * SIMD_CHUNK_SIZE);
                sum_chunks[u] = _mm512_fmadd_ps(vec1_chunk, vec2_chunk, sum_chunks[u]);
            }
        }

        #pragma GCC unroll 2
        for (std::size_t stride = 1; stride < unroll_size; stride *= 2) {
            #pragma GCC unroll 2
            for (std::size_t u = 0; u < unroll_size; u += stride * 2) {
                sum_chunks[u] = _mm512_add_ps(sum_chunks[u], sum_chunks[u + stride]);
            }
        }

        for (; i < NUM_SIMD_CHUNKS; ++i) {
            const __m512 vec1_chunk = _mm512_loadu_ps(vec1 + i * SIMD_CHUNK_SIZE);
            const __m512 vec2_chunk = _mm512_loadu_ps(vec2 + i * SIMD_CHUNK_SIZE);
            sum_chunks[0] = _mm512_fmadd_ps(vec1_chunk, vec2_chunk, sum_chunks[0]);
        }

        return static_cast<distance_t>(_mm512_reduce_add_ps(sum_chunks[0]));
    }


    __attribute__((always_inline))
    auto _impl_cosine(const vec_ele_t* vec1, const vec_ele_t* vec2) const -> distance_t {
        // True per-call cosine distance on raw (un-normalized) vectors:
        //   d = 1 - dot(a,b) / (||a|| * ||b||)
        // A single pass accumulates the cross term and both squared norms.
        // Vectors are SIMD-padded upstream (VectorDataset::_pad_to_simd_alignment),
        // and the zero tail contributes nothing to any of the three sums, so no
        // masked-tail path is needed. A single accumulator per quantity keeps the
        // register pressure bounded for the simdu2/simdu4 instantiations while the
        // three independent FMA chains still provide enough ILP.
        __m512 cross = _mm512_setzero_ps();   // Sum a*b
        __m512 nsq1  = _mm512_setzero_ps();   // Sum a*a = ||a||^2
        __m512 nsq2  = _mm512_setzero_ps();   // Sum b*b = ||b||^2

        for (std::size_t i = 0; i < NUM_SIMD_CHUNKS; ++i) {
            const __m512 a = _mm512_loadu_ps(vec1 + i * SIMD_CHUNK_SIZE);
            const __m512 b = _mm512_loadu_ps(vec2 + i * SIMD_CHUNK_SIZE);
            cross = _mm512_fmadd_ps(a, b, cross);
            nsq1  = _mm512_fmadd_ps(a, a, nsq1);
            nsq2  = _mm512_fmadd_ps(b, b, nsq2);
        }

        const float dot   = _mm512_reduce_add_ps(cross);
        const float sq1   = _mm512_reduce_add_ps(nsq1);   // ||a||^2
        const float sq2   = _mm512_reduce_add_ps(nsq2);   // ||b||^2
        // denom = ||a|| * ||b||, computed as sqrt(||a||^2 * ||b||^2) so a single
        // sqrt folds both magnitudes; it is the cosine denominator.
        const float denom = std::sqrt(sq1 * sq2);

        // Smaller = closer; result lies in [0, 2]. A zero-length vector has an
        // undefined cosine — treat it as maximally distant so it never wins a
        // min-heap.
        return denom > 0.0f ? static_cast<distance_t>(1.0f - dot / denom)
                            : static_cast<distance_t>(1.0f);
    }

};  // class SIMDDistance

}  // namespace cpu
}   // namespace artea
