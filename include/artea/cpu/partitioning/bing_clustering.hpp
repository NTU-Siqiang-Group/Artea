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
 * @FilePath: /Artea/include/artea/cpu/partitioning/bing_clustering.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2025-11-26 19:54:42
 * @Date: 2025-11-26 19:54:02
 * @Description: Bing Clustering Algorithm for streaming vector data partitioning.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <utility>
#include <limits>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <memory>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include <artea/cpu/containers/vector_array.hpp>
#include <artea/cpu/partitioning/stream_clustering.hpp>

// TODO : Implement BingClustering class
namespace artea {
namespace cpu {

/** @brief Bing (Balanced and Incremental Clustering using Navigable Graph) Algorithm for streaming vector data partitioning.
  * @tparam vertex_num_t Type for vertex indices.
  * @tparam vec_ele_t   Type for vector elements.
  * @tparam dist_func_t Type for the distance function functor.
 */
template <
    typename vertex_num_t,
    typename vec_ele_t,
    typename dist_func_t
>
class BingClustering :
    public StreamClustering<vertex_num_t, vec_ele_t, dist_func_t,
        BingClustering<vertex_num_t, vec_ele_t, dist_func_t>>
{

    using vertex_id_t = vertex_num_t;
    using distance_t = vec_ele_t;
    using base_class_t = StreamClustering<vertex_num_t, vec_ele_t, dist_func_t,
        BingClustering<vertex_num_t, vec_ele_t, dist_func_t>>;

public:
    BingClustering(
        const vec_dim_t vec_dim,
        const dist_func_t& dist_func
    ) : base_class_t(vec_dim, dist_func)
    {}

protected:


};  // class BingClustering

}   // namespace cpu
}   // namespace artea
