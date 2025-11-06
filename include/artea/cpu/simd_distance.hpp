#pragma once

#include <cstddef>
#include <cstdint>
#include <immintrin.h>
#include <cassert>

#include <artea/types.hpp>
#include <artea/logger.hpp>

namespace artea {
namespace cpu {

enum class DistanceMetrics {
    EUCLIDEAN,
    DOT,
    COSINE,
};

template <
    typename vec_ele_t,
    DistanceMetrics dist_type = DistanceMetrics::EUCLIDEAN,
    uint32_t unroll_size = 1
>
class SIMDDistance {

    static constexpr std::size_t SIMD_REGISTER_BITS = 512;
    static constexpr std::size_t SIMD_REGISTER_BYTES = SIMD_REGISTER_BITS / 8;  // 64
    // Number of elements that can be processed in a single SIMD register
    // e.g. we can process 16 elements per chunk for float32 type
    static constexpr std::size_t SIMD_CHUNK_SIZE = [] {
        static_assert(
            SIMD_REGISTER_BYTES % sizeof(vec_ele_t) == 0,
            "SIMD register size must be a multiple of the element size for this utility."
        );
        return SIMD_REGISTER_BYTES / sizeof(vec_ele_t);
    }();

public: 

    SIMDDistance(const vec_dim_t vec_dim) : _vec_dim(vec_dim), NUM_SIMD_CHUNKS(_vec_dim / SIMD_CHUNK_SIZE), NUM_REMAINING_ELES(_vec_dim % SIMD_CHUNK_SIZE) {
        ArteaLogger logger("SIMDDistance", LogLevel::INFO);
        if (vec_dim % SIMD_CHUNK_SIZE != 0) {
            logger.error(
                "Vector dimension must be a multiple of SIMD chunk size (e.g. 16 for float type and 8 for double type)"
            );
        }
    }

    __attribute__((always_inline))
    auto operator()(const vec_ele_t* vec1, const vec_ele_t* vec2) -> vec_ele_t {
        if constexpr (dist_type == DistanceMetrics::EUCLIDEAN) {
            return _impl_euclidean(vec1, vec2);
        } else if constexpr (dist_type == DistanceMetrics::DOT) {
            return _impl_dot(vec1, vec2);
        } else if constexpr (dist_type == DistanceMetrics::COSINE) {
            return _impl_cosine(vec1, vec2);
        } else {
            throw std::runtime_error("Invalid DistanceMetrics");
        }
    }

private:

    const vec_dim_t _vec_dim;
    /** @brief Number of SIMD chunks that can be processed in parallel */
    const std::size_t NUM_SIMD_CHUNKS;      
    /** @brief Number of remaining elements that cannot be processed in parallel */
    const std::size_t NUM_REMAINING_ELES;   

    __attribute__((always_inline))
    auto _impl_euclidean(const vec_ele_t* vec1, const vec_ele_t* vec2) -> vec_ele_t {
        // AVX512 implementation
        vec_ele_t result_chunk [NUM_SIMD_CHUNKS] __attribute__((aligned(64)));
        __m512 vec1_chunk, vec2_chunk, diff_chunk, sum_chunk = _mm512_set1_ps(0.0f);

        // Process SIMD chunks

        // TODO: unroll the loop
        if constexpr (unroll_size == 1) {
            for (std::size_t i = 0; i < NUM_SIMD_CHUNKS; ++i) {
                vec1_chunk = _mm512_loadu_ps(vec1 + i * SIMD_CHUNK_SIZE);
                vec2_chunk = _mm512_loadu_ps(vec2 + i * SIMD_CHUNK_SIZE);
                diff_chunk = _mm512_sub_ps(vec1_chunk, vec2_chunk);
                // sum_chunk = _mm512_add_ps(sum_chunk, _mm512_mul_ps(diff_chunk, diff_chunk));
                sum_chunk = _mm512_fmadd_ps(diff_chunk, diff_chunk, sum_chunk);
            }
        }
        else if constexpr (unroll_size == 2) {

        }
        else if constexpr (unroll_size == 4) {
            
        }

        // Process remaining elements
        if (NUM_REMAINING_ELES > 0) {
            throw std::runtime_error(
                "Currently vector dimension must be a multiple of SIMD chunk size (e.g. 16 for float type)"
            );
        }

        // // Horizontal sum
        // _mm512_storeu_ps(result_chunk, sum_chunk);
        // vec_ele_t result = 0.0f;

        // // TODO: unroll the loop
        // for (std::size_t i = 0; i < SIMD_CHUNK_SIZE; ++i) {
        //     result += result_chunk[i];
        // }

        // return result;

        return _mm512_reduce_add_ps(sum_chunk);
    }

    __attribute__((always_inline))
    auto _impl_dot(const vec_ele_t* vec1, const vec_ele_t* vec2) -> vec_ele_t {
        throw std::runtime_error("Currently DOT distance is not supported");
    }


    __attribute__((always_inline))
    auto _impl_cosine(const vec_ele_t* vec1, const vec_ele_t* vec2) -> vec_ele_t {
        throw std::runtime_error("Currently COSINE distance is not supported");
    }

};  // class SIMDDistance
    
}  // namespace cpu
}   // namespace artea