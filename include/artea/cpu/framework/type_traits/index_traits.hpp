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
template <typename IndexTraitsT, typename DerivedClassT> class DescentGraph;
template <typename IndexTraitsT, typename DerivedClassT, typename LayerGraphT> class HierarchicalGraph;
namespace conv_graph {
    template <typename IndexTraitsT> class IndexStructure;
}
namespace knn_graph {
    template <typename IndexTraitsT>
    using IndexStructure = conv_graph::IndexStructure<IndexTraitsT>;
}
namespace symmetric_knn_graph {
    template <typename IndexTraitsT>
    using IndexStructure = conv_graph::IndexStructure<IndexTraitsT>;
}
namespace artea_graph {
    template <typename IndexTraitsT> class IndexStructure;
}
template <typename IndexTraitsT> class CompactDescentGraph;
template <typename IndexTraitsT> class HierarchicalSearchGraph;
template <typename IndexTraitsT> class InterLayerLinks;
template <typename IndexTraitsT> class HierarchyManager;
template <typename IndexTraitsT> class DescentGraphCompactor;
template <typename IndexTraitsT> class FlatGraphFileManager;
template <typename IndexTraitsT> class HierarchicalGraphFileManager;
template <typename IndexTraitsT> class IndexSizeCalculator;
template <typename IndexTraitsT> class RadiusProber;
template <typename IndexTraitsT> class CompactInternalGraph;
template <typename IndexTraitsT> class InternalGraph;
template <typename IndexTraitsT> class InternalGraphCompactor;
template <typename IndexTraitsT> class HierarchicalGraphV2;
template <typename IndexTraitsT> class StackedRGraph;

template <typename BaseTraitsT>
struct IndexTraits : virtual public BaseTraitsT {

    /** ------ Self Traits ------ **/
    using index_traits_t = IndexTraits<BaseTraitsT>;

    /** @brief Base traits type. */
    using base_traits_t = BaseTraitsT;

    /** @brief CRTP base descent graph type (template on DerivedClassT). */
    template <typename DerivedClassT>
    using descent_graph_t = DescentGraph<index_traits_t, DerivedClassT>;

    /** @brief Flat search graph type (CSR format). */
    using compact_descent_graph_t = CompactDescentGraph<index_traits_t>;

    /** @brief CRTP base hierarchical graph type (template on DerivedClassT and LayerGraphT). */
    template <typename DerivedClassT, typename LayerGraphT>
    using hierarchical_graph_t = HierarchicalGraph<index_traits_t, DerivedClassT, LayerGraphT>;

    /** @brief Namespace-scoped index types for conv_graph, extending BaseTraits::conv_graph. */
    struct conv_graph : BaseTraitsT::conv_graph {
        conv_graph() = delete;
        using index_t = cpu::conv_graph::IndexStructure<index_traits_t>;
    };

    /** @brief Namespace-scoped index types for knn_graph, extending BaseTraits::knn_graph. */
    struct knn_graph : BaseTraitsT::knn_graph {
        knn_graph() = delete;
        using index_t = cpu::knn_graph::IndexStructure<index_traits_t>;
    };

    /** @brief Namespace-scoped index types for symmetric_knn_graph, extending BaseTraits::symmetric_knn_graph. */
    struct symmetric_knn_graph : BaseTraitsT::symmetric_knn_graph {
        symmetric_knn_graph() = delete;
        using index_t = cpu::symmetric_knn_graph::IndexStructure<index_traits_t>;
    };

    /** @brief Namespace-scoped index types for artea_graph, extending BaseTraits::artea_graph. */
    struct artea_graph : BaseTraitsT::artea_graph {
        artea_graph() = delete;
        using index_t = cpu::artea_graph::IndexStructure<index_traits_t>;
    };

    /** @brief Hierarchical search graph type. */
    using hierarchical_search_graph_t = HierarchicalSearchGraph<index_traits_t>;

    /** @brief Inter-layer links type. */
    using inter_layer_links_t = InterLayerLinks<index_traits_t>;

    /** @brief Hierarchical vector manager type. */
    using hierarchy_manager_t = HierarchyManager<index_traits_t>;

    /** @brief Descent graph compactor type. */
    using descent_graph_compactor_t = DescentGraphCompactor<index_traits_t>;

    /** @brief Flat graph file manager type. */
    using flat_graph_file_manager_t = FlatGraphFileManager<index_traits_t>;

    /** @brief Hierarchical graph file manager type. */
    using hierarchical_graph_file_manager_t = HierarchicalGraphFileManager<index_traits_t>;

    /** @brief Index size calculator type. */
    using index_size_calculator_t = IndexSizeCalculator<index_traits_t>;

    /** @brief Radius prober type. */
    using radius_prober_t = RadiusProber<index_traits_t>;

    /** @brief Compact internal graph type. */
    using compact_internal_graph_t = CompactInternalGraph<index_traits_t>;

    /** @brief Internal graph type (supports concurrent vertex/neighbor insertion). */
    using internal_graph_t = InternalGraph<index_traits_t>;

    /** @brief Single-layer compactor: InternalGraph -> CompactInternalGraph. */
    using internal_graph_compactor_t = InternalGraphCompactor<index_traits_t>;

    /** @brief Hierarchical graph V2 type (holds InternalGraph layers, atomic lnbr_t entry point). */
    using hierarchical_graph_v2_t = HierarchicalGraphV2<index_traits_t>;

    /** @brief Stacked R-net graph type (dynamic insertion over HierarchicalGraphV2). */
    using stacked_rgraph_t = StackedRGraph<index_traits_t>;

    /** @brief Minimum number of vertices required for a layer to continue building upper layers. */
    static constexpr uint32_t min_num_layer_vertex = 1024;

};  // struct IndexTraits

}   // namespace cpu
}   // namespace artea
