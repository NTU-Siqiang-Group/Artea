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

#include <artea/common/definitions.hpp>
#include <artea/cpu/router/vector_router.hpp>

namespace artea {
namespace cpu {

template <typename ComputerTraitsT, bool IntraQueryParallel = false>
class ProximityGraphRouter : public VectorRouter<ComputerTraitsT, ProximityGraphRouter<ComputerTraitsT, IntraQueryParallel>>
{

    using vec_num_t = typename ComputerTraitsT::vec_num_t;
    using vec_id_t = typename ComputerTraitsT::vec_id_t;
    using vec_ele_t = typename ComputerTraitsT::vec_ele_t;
    using distance_t = typename ComputerTraitsT::distance_t;
    using dist_func_t = typename ComputerTraitsT::dist_func_t;
    using vector_array_t = typename ComputerTraitsT::vector_array_t;
    using base_vecs_t = typename ComputerTraitsT::base_vecs_t;
    using base_class_t = VectorRouter<ComputerTraitsT, ProximityGraphRouter<ComputerTraitsT, IntraQueryParallel>>;

public:

    ProximityGraphRouter(
        const base_vecs_t& base_vecs,
        const dist_func_t& dist_func
    ) : base_class_t(base_vecs, dist_func)
    {}

    auto initialize_impl() -> void {
        // Do nothing
    }

    /**
     * @brief Query implementation for proximity graph router.
     */
    auto query_impl(const vec_ele_t* query_vec) const -> vec_id_t {
    }

};  // class ProximityGraphRouter

}   // namespace cpu
}   // namespace artea