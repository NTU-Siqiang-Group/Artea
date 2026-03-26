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

/**
 * @brief Policy for vertex generation strategy.
 */
enum class VGPolicyT {
    random_selection,  ///< Random vertex selection
    rnet_selection     ///< R-net based greedy selection
};

/**
 * @brief Policy for edge generation strategy.
 */
enum class EGPolicyT {
    conv_graph_descent,                 ///< Convergent graph descent
    speculative_conv_graph_descent      ///< Speculative convergent graph descent
};

/** ------ Forward Declaration  ------ **/
template <typename GraphFactoryTraitsT, typename DerivedClassT> class FlatGraphFactory;
template <typename GraphFactoryTraitsT, typename DerivedClassT> class HierarchicalGraphFactory;
template <typename GraphFactoryTraitsT> class ConvGraphFactory;
template <typename GraphFactoryTraitsT> class HierarchicalVerticesBuilder;
template <typename GraphFactoryTraitsT> class HierarchicalEdgesBuilder;
template <typename GraphFactoryTraitsT> class RoutingUpdater;
template <typename GraphFactoryTraitsT> class ArteaGraphFactory;

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

    /** @brief Vertex generation policy type. */
    using vg_policy_t = VGPolicyT;

    /** @brief Edge generation policy type. */
    using eg_policy_t = EGPolicyT;

    /** @brief Type for graph factory. */
    template <typename DerivedClassT>
    using flat_graph_factory_t = FlatGraphFactory<graph_factory_traits_t, DerivedClassT>;

    template <typename DerivedClassT>
    using hierarchical_graph_factory_t = HierarchicalGraphFactory<graph_factory_traits_t, DerivedClassT>;

    using conv_graph_factory_t = ConvGraphFactory<graph_factory_traits_t>;

    using hierarchical_vertices_builder_t = HierarchicalVerticesBuilder<graph_factory_traits_t>;

    using hierarchical_edges_builder_t = HierarchicalEdgesBuilder<graph_factory_traits_t>;

    using artea_graph_factory_t = ArteaGraphFactory<graph_factory_traits_t>;

};  // struct GraphFactoryTraits

}   // namespace cpu
}   // namespace artea
