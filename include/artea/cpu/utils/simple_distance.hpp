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
 * @FilePath: /Artea/include/artea/cpu/utils/simple_distance.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Simple distance functions for low-dimensional vectors (no SIMD)
 */

#pragma once

#include <cmath>
#include <cstdint>

namespace artea {
namespace cpu {

/**
 * @brief Simple Euclidean distance function for low-dimensional vectors.
 *
 * This class provides a basic Euclidean distance calculation without SIMD optimizations.
 * It's suitable for testing and for low-dimensional vectors (e.g., 2D, 3D) where SIMD
 * overhead would not provide benefits.
 *
 * @tparam ComputerTraitsT The computer traits type
 * @tparam Dummy Dummy parameter for template compatibility (unused)
 */
template <typename ComputerTraitsT, int Dummy = 0>
class SimpleEuclideanDistance {
public:
    using vec_dim_t = typename ComputerTraitsT::vec_dim_t;
    using vec_ele_t = typename ComputerTraitsT::vec_ele_t;
    using distance_t = typename ComputerTraitsT::distance_t;

    explicit SimpleEuclideanDistance(vec_dim_t dim) : dim_(dim) {}

    /**
     * @brief Compute Euclidean distance between two vectors.
     *
     * Uses #pragma omp simd for basic vectorization on supported compilers,
     * but does not require AVX-512 or specific SIMD instruction sets.
     */
    __attribute__((always_inline))
    distance_t operator()(const vec_ele_t* vec1, const vec_ele_t* vec2) const {
        distance_t sum = 0.0f;
        #pragma omp simd reduction(+:sum)
        for (vec_dim_t i = 0; i < dim_; ++i) {
            distance_t diff = static_cast<distance_t>(vec1[i]) - static_cast<distance_t>(vec2[i]);
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }

private:
    vec_dim_t dim_;
};

}   // namespace cpu
}   // namespace artea