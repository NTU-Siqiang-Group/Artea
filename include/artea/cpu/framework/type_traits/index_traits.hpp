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
template <typename IndexTraitsT, typename DerivedClassT> class RefiningGraph;
namespace conv_graph {
    template <typename IndexTraitsT> class IndexStructure;
}
namespace knn_graph {
    template <typename IndexTraitsT> class IndexStructure;
}
namespace symmetric_knn_graph {
    template <typename IndexTraitsT>
    using IndexStructure = knn_graph::IndexStructure<IndexTraitsT>;
}
namespace compact {
    template <typename IndexTraitsT> class RefiningGraph;
    template <typename IndexTraitsT> class HierarchicalGraph;
}
namespace dynamic {
    template <typename IndexTraitsT, typename DerivedClassT> class RefiningGraph;
    template <typename IndexTraitsT> class HierarchicalGraph;
}
template <typename IndexTraitsT> class RefiningGraphCompactor;
template <typename IndexTraitsT> class FlatGraphFileManager;
template <typename IndexTraitsT> class IndexSizeCalculator;
template <typename IndexTraitsT> class RadiusProber;
template <typename IndexTraitsT> class HierarchicalGraphCompactor;
namespace stacked_rgraph {
    template <typename IndexTraitsT> class IndexStructure;
}

template <typename BaseTraitsT>
struct IndexTraits : virtual public BaseTraitsT {

    /** ------ Self Traits ------ **/
    using index_traits_t = IndexTraits<BaseTraitsT>;

    /** @brief Base traits type. */
    using base_traits_t = BaseTraitsT;

    /** @brief CRTP base descent graph type (template on DerivedClassT). */
    template <typename DerivedClassT>
    using refining_graph_t = dynamic::RefiningGraph<index_traits_t, DerivedClassT>;

    /** @brief Graph structures grouped by mode. */
    struct compact {
        compact() = delete;
        using refining_graph_t       = cpu::compact::RefiningGraph<index_traits_t>;
        using hierarchical_graph_t = cpu::compact::HierarchicalGraph<index_traits_t>;
    };
    struct dynamic {
        dynamic() = delete;
        using hierarchical_graph_t = cpu::dynamic::HierarchicalGraph<index_traits_t>;
    };

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

    /** @brief Descent graph compactor type. */
    using refining_graph_compactor_t = RefiningGraphCompactor<index_traits_t>;

    /** @brief Flat graph file manager type. */
    using flat_graph_file_manager_t = FlatGraphFileManager<index_traits_t>;

    /** @brief Index size calculator type. */
    using index_size_calculator_t = IndexSizeCalculator<index_traits_t>;

    /** @brief Radius prober type. */
    using radius_prober_t = RadiusProber<index_traits_t>;

    /** @brief Multi-layer compactor: dynamic::HierarchicalGraph -> compact::HierarchicalGraph. */
    using hierarchical_graph_compactor_t = HierarchicalGraphCompactor<index_traits_t>;

    /** @brief Namespace-scoped index types for stacked_rgraph
     *  (dynamic r-net insertion over HierarchicalGraph). */
    struct stacked_rgraph : BaseTraitsT::stacked_rgraph {
        stacked_rgraph() = delete;
        using index_t = cpu::stacked_rgraph::IndexStructure<index_traits_t>;
    };

    /** @brief Minimum number of vertices required for a layer to continue building upper layers. */
    static constexpr uint32_t min_num_layer_vertex = 1024;

};  // struct IndexTraits

}   // namespace cpu
}   // namespace artea
