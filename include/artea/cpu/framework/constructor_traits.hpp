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
 * @FilePath: /Artea/include/artea/cpu/framework/constructor_traits.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description:
 */

#pragma once

namespace artea {
namespace cpu {

/** ------ Forward Declaration  ------ **/
template <typename ConstructorTraitsT> class PropagateEngine;
template <typename ConstructorTraitsT> class GraphConstructor;

template <
    typename VertexGeneratorTraitsT,
    typename EdgeGeneratorTraitsT,
    typename IndexTraitsT,
    bool SelectiveSchedule = false
>
struct ConstructorTraits :
    public VertexGeneratorTraitsT,
    public EdgeGeneratorTraitsT
{
    using contructor_traits_t = ConstructorTraits<VertexGeneratorTraitsT, EdgeGeneratorTraitsT, IndexTraitsT, SelectiveSchedule>;

    /** @brief Type for graph constructor. */
    using graph_constructor_t = GraphConstructor<contructor_traits_t>;

    /** @brief Type for propagation engine. */
    using propagate_engine_t = PropagateEngine<contructor_traits_t>;

    /** @brief Indicates whether to enable selective scheduling. */
    static constexpr bool selective_schedule = SelectiveSchedule;

};  // struct ConstructorTraits

}   // namespace cpu
}   // namespace artea
