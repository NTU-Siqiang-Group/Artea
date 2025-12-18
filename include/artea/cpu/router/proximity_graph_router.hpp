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
 * @FilePath: /Artea/include/artea/cpu/router/proximity_graph_router.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description:
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <functional>

namespace artea {
namespace cpu {

template <typename RouterTraitsT>
class ProximityGraphRouter : public RouterTraitsT::cluster_router_t
{

    using vertex_num_t = typename RouterTraitsT::vertex_num_t;
    using vertex_id_t = typename RouterTraitsT::vertex_id_t;
    using vec_ele_t = typename RouterTraitsT::vec_ele_t;
    using distance_t = typename RouterTraitsT::distance_t;
    using dist_func_t = typename RouterTraitsT::dist_func_t;
    using vector_array_t = typename RouterTraitsT::vector_array_t;
    using cluster_id_t = typename RouterTraitsT::cluster_id_t;
    static constexpr bool intra_query_parallel = RouterTraitsT::intra_query_parallel;
    using base_class_t = typename RouterTraitsT::cluster_router_t;

public:

    ProximityGraphRouter(
        const vector_array_t& centroids,
        const dist_func_t& dist_func
    ) : base_class_t(centroids, dist_func)
    {}

    auto initialize_impl() -> void {
        // Do nothing
    }

    /**
     * @brief Query implementation for proximity graph router.
     */
    auto query_impl(const vec_ele_t* query_vec) const -> cluster_id_t {

    }

};  // class ProximityGraphRouter

}   // namespace cpu
}   // namespace artea