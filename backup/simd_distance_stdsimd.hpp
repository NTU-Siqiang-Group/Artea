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
#include <experimental/simd>
#include <cassert>
#include <artea/common/logger.hpp>

namespace stdx = std::experimental;

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

    using simd_t = std::experimental::native_simd<vec_ele_t>;
    static constexpr std::size_t W = simd_t::size();

public:

    SIMDDistance(const vec_dim_t vec_dim) : _vec_dim(vec_dim) {}

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

    __attribute__((always_inline))
    auto _impl_euclidean(const vec_ele_t* vec1, const vec_ele_t* vec2) const -> distance_t {
        constexpr std::size_t stride = W * unroll_size;
        simd_t sums[unroll_size] = {};
        std::size_t i = 0;

        // Main unrolled loop
        for (; i + stride <= _vec_dim; i += stride) {
            for (std::size_t u = 0; u < unroll_size; ++u) {
                simd_t va(vec1 + i + u * W, stdx::element_aligned);
                simd_t vb(vec2 + i + u * W, stdx::element_aligned);
                simd_t diff = va - vb;
                sums[u] += diff * diff;
            }
        }

        // Tail: remaining full SIMD lanes
        for (; i + W <= _vec_dim; i += W) {
            simd_t va(vec1 + i, stdx::element_aligned);
            simd_t vb(vec2 + i, stdx::element_aligned);
            simd_t diff = va - vb;
            sums[0] += diff * diff;
        }

        // Merge accumulators
        for (std::size_t u = 1; u < unroll_size; ++u) sums[0] += sums[u];
        distance_t result = stdx::reduce(sums[0]);

        // Scalar tail for remaining elements
        for (; i < _vec_dim; ++i) {
            vec_ele_t diff = vec1[i] - vec2[i];
            result += diff * diff;
        }

        return result;
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