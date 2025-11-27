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
 * @FilePath: /Artea/include/artea/cpu/partitioning/stream_clustering.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description:
 */

namespace artea {
namespace cpu {

template <
    typename vertex_num_t,
    typename vec_ele_t,
    typename dist_func_t,
    typename derived_class_t
>
class StreamClustering {

public:
    StreamClustering(
        const vec_dim_t vec_dim,
        const dist_func_t& dist_func
    ) :
        _vec_dim(vec_dim),
        _dist_func(dist_func)
    {}

protected:

    /** @brief Dimension of each vector. */
    const vec_dim_t _vec_dim;

    /** @brief Reference to the injected distance function functor. */
    const dist_func_t& _dist_func;

};

}   // namespace cpu
}   // namespace artea