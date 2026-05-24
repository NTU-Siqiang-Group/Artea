// Copyright 2025 yeweitang
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
 * @FilePath: /Artea/include/artea/cpu/partitioning/vector_router.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description:
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <functional>
#include <utility>

namespace artea {
namespace cpu {

template <typename RouterTraitsT, typename DistFuncT, typename DerivedClassT>
class VectorRouter {

    using vec_id_t = typename RouterTraitsT::vec_id_t;
    using vec_num_t = typename RouterTraitsT::vec_num_t;
    using vec_ele_t = typename RouterTraitsT::vec_ele_t;
    using distance_t = typename RouterTraitsT::distance_t;
    using dist_func_t = DistFuncT;
    using vector_array_t = typename RouterTraitsT::vector_array_t;
    using query_vecs_t = typename RouterTraitsT::query_vecs_t;
    using knn_results_t = typename RouterTraitsT::knn_results_t;

public:

    VectorRouter(
        const vector_array_t& vecs_data,
        const dist_func_t& dist_func,
        const uint32_t topk
    ) :
        _num_vecs(vecs_data.get_num_vecs()),
        _vecs_data(vecs_data),
        _dist_func(dist_func),
        _topk(topk)
    {}

protected:

    /** @brief Number of vectors in the dataset. */
    const vec_num_t _num_vecs;

    /** @brief Reference to the vector data. */
    const vector_array_t& _vecs_data;

    /** @brief Reference to the injected distance function functor. */
    const dist_func_t& _dist_func;

    /** @brief Number of nearest neighbors to return. */
    const uint32_t _topk;

};  // class VectorRouter

}   // namespace cpu
}   // namespace artea
