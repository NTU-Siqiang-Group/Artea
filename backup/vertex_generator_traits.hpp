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
 * @FilePath: /Artea/include/artea/cpu/framework/vertex_generator_traits.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description:
 */

#pragma once

namespace artea {
namespace cpu {

/** ------ Forward Declaration  ------ **/
template <typename VertexGeneratorTraitsT, typename DerivedClassT> class VertexGenerator;
template <typename VertexGeneratorTraitsT> class RandomVG;

template <typename ComputerTraitsT, typename IndexTraitsT>
struct VertexGeneratorTraits :
    virtual public ComputerTraitsT,
    virtual public IndexTraitsT
{

    /** ------ Self Traits ------ **/
    using vertex_generator_traits_t = VertexGeneratorTraits<ComputerTraitsT, IndexTraitsT>;

    /** @brief Vertex subset result type. */
    using vertex_subset_t = typename ComputerTraitsT::vertex_subset_t;

    /** @brief Random vertex generator. */
    using random_vg_t = RandomVG<vertex_generator_traits_t>;

    template <typename DerivedClassT>
    using vertex_generator_t = VertexGenerator<vertex_generator_traits_t, DerivedClassT>;

};  // struct VertexGeneratorTraits

}   // namespace cpu
}   // namespace artea
