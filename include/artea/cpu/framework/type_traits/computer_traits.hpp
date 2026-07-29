// Copyright 2025 Weitang Ye
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
 * @FilePath: /Artea/include/artea/cpu/framework/computer_traits.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description:
 */

#pragma once

#include <cstddef>
#include <vector>
#include <utility>

namespace artea {
namespace cpu {

/** ------ Forward Declaration  ------ **/
template <typename ComputerTraitsT, std::size_t UnrollSize> class SIMDDistance;
template <typename ComputerTraitsT, std::size_t UnrollSize> class SIMDLinear;
template <typename ComputerTraitsT> class RecallEstimator;
template <typename ComputerTraitsT> class ADREstimator;
template <typename ComputerTraitsT> class DistanceProber;
template <typename ComputerTraitsT> class DatasetProber;

/** @brief Distance metrics used for computing distances between vectors */
enum class DistanceMetricsT : uint8_t {
    EUCLIDEAN,
    DOT,
    COSINE
};  // enum class DistanceMetricsT

/** @brief Traits for computing distances between vectors.
 *
 *  The vector dimension @p VecDim is a compile-time template parameter so the
 *  SIMD distance/linear kernels know their chunk count statically and no
 *  longer take a runtime dimension via their constructor. @p VecDim must be the
 *  SIMD-padded dimension (a multiple of the SIMD chunk size). */
template <typename BaseTraitsT, DistanceMetricsT DistanceMetrics, typename BaseTraitsT::vec_dim_t VecDim>
struct ComputerTraits : virtual public BaseTraitsT {

private:

    /** ------ Self Traits ------ **/
    using computer_traits_t = ComputerTraits<BaseTraitsT, DistanceMetrics, VecDim>;

public:

    /** @brief Base traits type. */
    using base_traits_t = BaseTraitsT;

    using distance_metrics_t = DistanceMetricsT;

    /** @brief Compile-time SIMD-padded vector dimension. */
    using vec_dim_t = typename BaseTraitsT::vec_dim_t;
    static constexpr vec_dim_t vec_dim = VecDim;

    // Distance function type selection
    template <std::size_t UnrollSize = 1>
    using simd_dist_t = SIMDDistance<computer_traits_t, UnrollSize>;

    using simdu1_dist_t = SIMDDistance<computer_traits_t, 1>;
    using simdu2_dist_t = SIMDDistance<computer_traits_t, 2>;
    using simdu4_dist_t = SIMDDistance<computer_traits_t, 4>;
    using dist_func_t = SIMDDistance<computer_traits_t, 1>;

    template <std::size_t UnrollSize = 1>
    using simd_linear_t = SIMDLinear<computer_traits_t, UnrollSize>;
    using simdu1_linear_t = SIMDLinear<computer_traits_t, 1>;
    using simdu2_linear_t = SIMDLinear<computer_traits_t, 2>;
    using simdu4_linear_t = SIMDLinear<computer_traits_t, 4>;
    using linear_func_t = SIMDLinear<computer_traits_t, 1>;

    using recall_estimator_t = RecallEstimator<computer_traits_t>;
    using adr_estimator_t = ADREstimator<computer_traits_t>;

    using distance_prober_t = DistanceProber<computer_traits_t>;
    using dataset_prober_t = DatasetProber<computer_traits_t>;

    static constexpr distance_metrics_t distance_metrics = DistanceMetrics;

};  // struct ComputerTraits

}   // namespace cpu
}   // namespace artea
