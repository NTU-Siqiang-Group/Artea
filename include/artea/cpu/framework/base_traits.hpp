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
 * @FilePath: /Artea/include/artea/cpu/framework/base_traits.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description:
 */

#pragma once

#include <cstddef>
#include <vector>
#include <utility>

#include <artea/cpu/containers/vector_array.hpp>

namespace artea {
namespace cpu {

/* ------ Forward Declarations ------ */
template <typename BaseTraitsT> struct Neighbor;
template <typename BaseTraitsT> class NbrLogTable;
template <typename BaseTraitsT> class VectorDataset;
template <typename BaseTraitsT> class SIMDDistance;
template <typename BaseTraitsT> class VectorDataset;

/* ------ Base Traits Definition ------ */
template <typename VertexNumT, typename VecEleT>
struct BaseTraits {

private:
    /** ------ Basic Type ------ **/
    using base_traits_t = BaseTraits<VertexNumT, VecEleT>;

public:
    /** @brief Type for vector dimensions. */
    using vec_dim_t = uint32_t;

    /** @brief Type for vertex numbers. */
    using vertex_num_t = VertexNumT;

    /** @brief Type for vertex identifiers. */
    using vertex_id_t = VertexNumT;

    /** @brief Type for number of vectors. */
    using vec_num_t = VertexNumT;

    /** @brief Type for vector identifiers. */
    using vec_id_t = VertexNumT;

    /** @brief Type for vector elements. */
    using vec_ele_t = VecEleT;

    /** @brief Type for distance values. */
    using distance_t = VecEleT;

    /** @brief Type for number of clusters. */
    using cluster_num_t = VertexNumT;

    /** @brief Type for cluster identifiers. */
    using cluster_id_t = VertexNumT;

    /** @brief Type for partition numbers. */
    using part_num_t = VertexNumT;

    /** @brief Type for partition identifiers. */
    using part_id_t = VertexNumT;

    /** @brief Type for iteration counts. */
    using iter_t = uint32_t;

    /** ------  ------ **/

    /** @brief Type for neighbor entries. */
    using nbr_t = Neighbor<base_traits_t>;

    /** @brief Type for neighbor arrays. */
    using nbr_arr_t = std::vector<nbr_t>;

    /** @brief Type for vector arrays. */
    using vector_array_t = VectorArray<vertex_num_t, vec_ele_t>;

    /** @brief Type for vector datasets. */
    using vector_dataset_t = VectorDataset<base_traits_t>;

    /** @brief Type for base/query vector arrays. */
    using base_queries_t = VectorArray<vertex_num_t, vec_ele_t>;

    /** @brief Type for ground truth vector arrays. */
    using groud_truth_t = VectorArray<vertex_num_t, vec_id_t>;

};  // struct BaseTraits

}   // namespace cpu
}   // namespace artea