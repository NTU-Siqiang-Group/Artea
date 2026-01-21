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

#include <artea/cpu/framework/base_traits.hpp>

namespace artea {
namespace cpu {

/** ------ Forward Declaration  ------ **/
template <typename ComputerTraitsT> class SIMDDistance;

/** @brief Distance metrics used for computing distances between vectors */
enum class distance_metrics_t {
    EUCLIDEAN,
    DOT,
    COSINE
};  // enum class distance_metrics_t

/** @brief Traits for computing distances between vectors */
template <
    typename BaseTraitsT,
    distance_metrics_t DistanceMetrics,
    std::size_t UnrollSize = 1
>
struct ComputerTraits : virtual public BaseTraitsT {

private:

    /** ------ Self Traits ------ **/
    using computer_traits_t = ComputerTraits<BaseTraitsT, DistanceMetrics, UnrollSize>;

public:

    /** @brief Base traits type. */
    using base_traits_t = BaseTraitsT;

    /** @brief Type for distance values. */
    using dist_func_t = SIMDDistance<computer_traits_t>;

    static constexpr distance_metrics_t distance_metrics = DistanceMetrics;

    static constexpr std::size_t unroll_size = UnrollSize;

};  // struct ComputerTraits

}   // namespace cpu
}   // namespace artea