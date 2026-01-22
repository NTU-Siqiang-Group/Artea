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
 * @FilePath: /Artea/include/artea/cpu/framework/router_traits.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description:
 */

#pragma once

namespace artea {
namespace cpu {

/** ------ Forward Declaration  ------ **/
template <typename RouterTraitsT, typename DerivedClassT> class VectorRouter;
template <typename RouterTraitsT> class BruteforceRouter;
template <typename RouterTraitsT> class ProximityGraphRouter;

/** @brief Traits for routing to queried vectors */
template <typename ComputerTraitsT, bool IntraQueryParallel = false>
struct RouterTraits : public ComputerTraitsT {

    /** ------ Self Traits ------ **/
    using router_traits_t = RouterTraits<ComputerTraitsT, IntraQueryParallel>;

    template <typename DerivedClassT>
    using vector_router_t = VectorRouter<router_traits_t, DerivedClassT>;

    /** @brief Type for bruteforce router. */
    using bruteforce_router_t = BruteforceRouter<router_traits_t>;

    /** @brief Type for proximity graph router. */
    using proximity_graph_router_t = ProximityGraphRouter<router_traits_t>;

    /** @brief Indicates whether to enable intra-query parallelism. */
    static constexpr bool intra_query_parallel = IntraQueryParallel;

};  // struct RouterTraits

}   // namespace cpu
}   // namespace artea