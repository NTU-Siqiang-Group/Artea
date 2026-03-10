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
 * @FilePath: /Artea/include/artea/cpu/framework/index_traits.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Traits for graph index structures.
 */

#pragma once

namespace artea {
namespace cpu {

/** ------ Forward Declaration  ------ **/
template <typename IndexTraitsT> class FlatGraph;
template <typename IndexTraitsT> class FlatSearchGraph;
template <typename IndexTraitsT> class HierarchicalGraph;
template <typename IndexTraitsT> class HierarchicalSearchGraph;
template <typename IndexTraitsT> class InterLayerLinks;
template <typename IndexTraitsT> class HierarchicalVecsManager;

template <typename BaseTraitsT>
struct IndexTraits : virtual public BaseTraitsT {

    /** ------ Self Traits ------ **/
    using index_traits_t = IndexTraits<BaseTraitsT>;

    /** @brief Base traits type. */
    using base_traits_t = BaseTraitsT;

    /** @brief Flat graph type. */
    using flat_graph_t = FlatGraph<index_traits_t>;

    /** @brief Flat search graph type (CSR format). */
    using flat_search_graph_t = FlatSearchGraph<index_traits_t>;

    /** @brief Hierarchical graph type. */
    using hierarchical_graph_t = HierarchicalGraph<index_traits_t>;

    /** @brief Hierarchical search graph type. */
    using hierarchical_search_graph_t = HierarchicalSearchGraph<index_traits_t>;

    /** @brief Inter-layer links type. */
    using inter_layer_links_t = InterLayerLinks<index_traits_t>;

    /** @brief Hierarchical vector manager type. */
    using hierarchical_vecs_manager_t = HierarchicalVecsManager<index_traits_t>;

};  // struct IndexTraits

}   // namespace cpu
}   // namespace artea