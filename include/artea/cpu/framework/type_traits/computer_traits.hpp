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
template <typename ComputerTraitsT, std::size_t VecDim, std::size_t UnrollSize = 1> class SIMDDistance;
template <typename ComputerTraitsT, std::size_t UnrollSize> class SIMDDistanceDispatcher;
template <typename ComputerTraitsT, std::size_t UnrollSize> class SIMDFMA;
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

/** @brief Traits for computing distances between vectors */
template <typename BaseTraitsT, DistanceMetricsT DistanceMetrics>
struct ComputerTraits : virtual public BaseTraitsT {

private:

    /** ------ Self Traits ------ **/
    using computer_traits_t = ComputerTraits<BaseTraitsT, DistanceMetrics>;

public:

    /** @brief Base traits type. */
    using base_traits_t = BaseTraitsT;

    using distance_metrics_t = DistanceMetricsT;

    // Distance function type selection.
    //
    // dist_func_t<VecDim>:
    //   Template alias for the static-dim SIMDDistance kernel. VecDim must
    //   be specified at the use site (e.g. typename CT::template dist_func_t<128>).
    //   Inside a std::visit lambda over a dispatcher's variant, the
    //   auto-deduced parameter type is exactly this alias for the chosen
    //   alternative — callers rarely need to spell the alias explicitly.
    //
    // simd_dispatcher_t:
    //   Concrete alias for the runtime-dim adapter. ctor takes vec_dim_t
    //   and selects one variant alternative; only exposes variant() for
    //   per-loop std::visit. This is what every consumer that previously
    //   held a dist_func_t now holds instead.
    template <std::size_t VecDim, std::size_t UnrollSize = 1>
    using dist_func_t = SIMDDistance<computer_traits_t, VecDim, UnrollSize>;

    using simd_dispatcher_t = SIMDDistanceDispatcher<computer_traits_t, 1>;

    template <std::size_t UnrollSize = 1>
    using simd_fma_t = SIMDFMA<computer_traits_t, UnrollSize>;
    using simdu1_fma_t = SIMDFMA<computer_traits_t, 1>;
    using simdu2_fma_t = SIMDFMA<computer_traits_t, 2>;
    using simdu4_fma_t = SIMDFMA<computer_traits_t, 4>;
    using fma_func_t = SIMDFMA<computer_traits_t, 1>;

    template <std::size_t UnrollSize = 1>
    using simd_linear_t = SIMDLinear<computer_traits_t, UnrollSize>;
    using simdu1_linear_t = SIMDLinear<computer_traits_t, 1>;
    using simdu2_linear_t = SIMDLinear<computer_traits_t, 2>;
    using simdu4_linear_t = SIMDLinear<computer_traits_t, 4>;
    using linear_func_t = SIMDLinear<computer_traits_t, 1>;

    template <std::size_t UnrollSize = 1>
    using fma_t = SIMDFMA<computer_traits_t, UnrollSize>;

    using recall_estimator_t = RecallEstimator<computer_traits_t>;
    using adr_estimator_t = ADREstimator<computer_traits_t>;

    using distance_prober_t = DistanceProber<computer_traits_t>;
    using dataset_prober_t = DatasetProber<computer_traits_t>;

    static constexpr distance_metrics_t distance_metrics = DistanceMetrics;

};  // struct ComputerTraits

}   // namespace cpu
}   // namespace artea