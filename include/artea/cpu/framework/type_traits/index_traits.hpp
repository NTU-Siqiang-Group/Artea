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
template <typename IndexTraitsT, typename DerivedClassT> class FlatGraph;
template <typename IndexTraitsT, typename DerivedClassT, typename LayerGraphT> class HierarchicalGraph;
namespace conv_graph {
    template <typename IndexTraitsT> class GraphIndex;
}
namespace artea_graph {
    template <typename IndexTraitsT, typename LayerGraphT> class GraphIndex;
}
template <typename IndexTraitsT> class FlatSearchGraph;
template <typename IndexTraitsT> class HierarchicalSearchGraph;
template <typename IndexTraitsT> class InterLayerLinks;
template <typename IndexTraitsT> class HierarchicalVecsManager;
template <typename IndexTraitsT> class SearchGraphConverter;
template <typename IndexTraitsT, typename FlatGraphT> class FlatGraphFileManager;
template <typename IndexTraitsT, typename HierGraphT, typename LayerGraphT> class HierarchicalGraphFileManager;
template <typename IndexTraitsT> class IndexSizeCalculator;

template <typename BaseTraitsT>
struct IndexTraits : virtual public BaseTraitsT {

    /** ------ Self Traits ------ **/
    using index_traits_t = IndexTraits<BaseTraitsT>;

    /** @brief Base traits type. */
    using base_traits_t = BaseTraitsT;

    /** @brief CRTP base flat graph type (template on DerivedClassT). */
    template <typename DerivedClassT>
    using flat_graph_t = FlatGraph<index_traits_t, DerivedClassT>;

    /** @brief Convergent graph index type (concrete, extends flat_graph_t). */
    using conv_graph_index_t = conv_graph::GraphIndex<index_traits_t>;

    /** @brief Flat search graph type (CSR format). */
    using flat_search_graph_t = FlatSearchGraph<index_traits_t>;

    /** @brief CRTP base hierarchical graph type (template on DerivedClassT and LayerGraphT). */
    template <typename DerivedClassT, typename LayerGraphT>
    using hierarchical_graph_t = HierarchicalGraph<index_traits_t, DerivedClassT, LayerGraphT>;

    /** @brief Artea hierarchical graph index type (concrete, extends hierarchical_graph_t). */
    template <typename LayerGraphT>
    using artea_graph_index_t = artea_graph::GraphIndex<index_traits_t, LayerGraphT>;

    /** @brief Hierarchical search graph type. */
    using hierarchical_search_graph_t = HierarchicalSearchGraph<index_traits_t>;

    /** @brief Inter-layer links type. */
    using inter_layer_links_t = InterLayerLinks<index_traits_t>;

    /** @brief Hierarchical vector manager type. */
    using hierarchical_vecs_manager_t = HierarchicalVecsManager<index_traits_t>;

    /** @brief Search graph converter type. */
    using search_graph_converter_t = SearchGraphConverter<index_traits_t>;

    /** @brief Flat graph file manager type (generic, parameterized by FlatGraphT). */
    template <typename FlatGraphT = conv_graph_index_t>
    using flat_graph_file_manager_t = FlatGraphFileManager<index_traits_t, FlatGraphT>;

    /** @brief Hierarchical graph file manager type (generic, parameterized by HierGraphT and LayerGraphT). */
    template <typename HierGraphT = artea_graph_index_t<conv_graph_index_t>, typename LayerGraphT = conv_graph_index_t>
    using hierarchical_graph_file_manager_t = HierarchicalGraphFileManager<index_traits_t, HierGraphT, LayerGraphT>;

    /** @brief Index size calculator type. */
    using index_size_calculator_t = IndexSizeCalculator<index_traits_t>;

    /** @brief Minimum number of vertices required for a layer to continue building upper layers. */
    static constexpr uint32_t min_num_layer_vertex = 1024;

};  // struct IndexTraits

}   // namespace cpu
}   // namespace artea
