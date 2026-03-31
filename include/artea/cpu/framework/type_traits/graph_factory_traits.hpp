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
 * @FilePath: /Artea/include/artea/cpu/framework/graph_factory_traits.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description:
 */

#pragma once

namespace artea {
namespace cpu {

/** ------ Forward Declaration  ------ **/
namespace conv_graph {
    template <typename GraphFactoryTraitsT> class IndexFactory;
}
namespace artea_graph {
    template <typename GraphFactoryTraitsT> class IndexFactory;
}

template <
    typename VertexGeneratorTraitsT,
    typename EdgeGeneratorTraitsT,
    typename RouterTraitsT
>
struct GraphFactoryTraits :
    public VertexGeneratorTraitsT,
    public EdgeGeneratorTraitsT
{
    using graph_factory_traits_t = GraphFactoryTraits<VertexGeneratorTraitsT, EdgeGeneratorTraitsT, RouterTraitsT>;

    /** @brief Convergent graph factory type. */
    using conv_graph_factory_t = conv_graph::IndexFactory<graph_factory_traits_t>;

    /** @brief Artea graph factory type. */
    using artea_graph_factory_t = artea_graph::IndexFactory<graph_factory_traits_t>;

};  // struct GraphFactoryTraits

}   // namespace cpu
}   // namespace artea
