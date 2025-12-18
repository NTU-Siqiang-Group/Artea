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
 * @FilePath: /Artea/include/artea/cpu/framework/router_traits.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description:
 */

#pragma once

#include <cstddef>
#include <vector>
#include <utility>

namespace artea {
namespace cpu {

/* ------ Forward Declarations ------ */
template <typename RouterTraitsT> class ClusterRouter;

template <typename RouterTraitsT> class BruteforceRouter;

template <typename RouterTraitsT> class ProximityGraphRouter;

/* ------ Router Definition ------ */

/**
 * @brief Router selection policies for cluster routing.
 */
enum class router_policy_t : uint8_t {
    BRUTEFORCE_ROUTER = 0,
    PROXIMITY_GRAPH_ROUTER = 1
};  // enum class router_policy_t

template <router_policy_t RouterPolicy, typename RouterTraitsT>
struct RouterImplSelector;

template <typename RouterTraitsT>
struct RouterImplSelector<router_policy_t::BRUTEFORCE_ROUTER, RouterTraitsT> {
    using type = BruteforceRouter<RouterTraitsT>;
};

template <typename RouterTraitsT>
struct RouterImplSelector<router_policy_t::PROXIMITY_GRAPH_ROUTER, RouterTraitsT> {
    using type = ProximityGraphRouter<RouterTraitsT>;
};

template <
    typename ComputerTraitsT,
    router_policy_t RouterPolicy,
    bool IntraQueryParallel
>
struct RouterTraits : public ComputerTraitsT {

private:

    /** ------ Self Traits ------ **/
    using router_traits_t = RouterTraits<ComputerTraitsT, RouterPolicy, IntraQueryParallel>;

public:

    /** @brief Type for vector elements. */
    using vec_ele_t = typename ComputerTraitsT::vec_ele_t;

    /** @brief Type for the specific router implementation. */
    using router_impl_t = typename RouterImplSelector<RouterPolicy,  router_traits_t>::type;

    /** @brief Type for the cluster router. */
    using cluster_router_t = ClusterRouter<router_traits_t>;

    /** @brief Whether to enable intra-query parallelism in cluster routing. */
    static constexpr bool intra_query_parallel = IntraQueryParallel;

};  // struct RouterTraits

}   // namespace cpu
}   // namespace artea