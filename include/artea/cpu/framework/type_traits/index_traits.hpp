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
    template <typename IndexTraitsT> class RefiningGraph;
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
namespace artea_graph {
    template <typename IndexTraitsT> class IndexStructure;
}

template <typename BaseTraitsT>
struct IndexTraits : virtual public BaseTraitsT {

    /** ------ Self Traits ------ **/
    using index_traits_t = IndexTraits<BaseTraitsT>;

    /** @brief Base traits type. */
    using base_traits_t = BaseTraitsT;

    /** @brief Floor on per-layer CSR capacity and minimum apex population
     *         for a top bucket to survive @c HierarchicalGraphCompactor
     *         trimming. Consumed by @c stacked_rgraph::RGraphConfig and
     *         @c HierarchicalGraphCompactor. */
    static constexpr typename BaseTraitsT::vertex_num_t min_layer_cap = 128;

    /** @brief Layer configuration type. */
    using layer_config_t = LayerConfig<index_traits_t>;

    /** @brief Graph structures grouped by mode. */
    struct compact {
        compact() = delete;
        using refining_graph_t     = cpu::compact::RefiningGraph<index_traits_t>;
        using hierarchical_graph_t = cpu::compact::HierarchicalGraph<index_traits_t>;
    };
    struct dynamic {
        dynamic() = delete;
        using refining_graph_t     = cpu::dynamic::RefiningGraph<index_traits_t>;
        using hierarchical_graph_t = cpu::dynamic::HierarchicalGraph<index_traits_t>;
    };

    /** @brief Namespace-scoped types for conv_graph. */
    struct conv_graph {
        conv_graph() = delete;
        using index_t            = cpu::conv_graph::IndexStructure<index_traits_t>;
        using propagate_config_t = cpu::conv_graph::PropagateConfig<index_traits_t>;
        using pruning_config_t   = cpu::conv_graph::PruningConfig<index_traits_t>;
    };

    /** @brief Namespace-scoped types for knn_graph. */
    struct knn_graph {
        knn_graph() = delete;
        using index_t            = cpu::knn_graph::IndexStructure<index_traits_t>;
        using propagate_config_t = cpu::knn_graph::PropagateConfig<index_traits_t>;
    };

    /** @brief Namespace-scoped types for symmetric_knn_graph. */
    struct symmetric_knn_graph {
        symmetric_knn_graph() = delete;
        using index_t            = cpu::symmetric_knn_graph::IndexStructure<index_traits_t>;
        using propagate_config_t = cpu::symmetric_knn_graph::PropagateConfig<index_traits_t>;
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

    /** @brief Namespace-scoped types for stacked_rgraph
     *  (dynamic r-net insertion over HierarchicalGraph). */
    struct stacked_rgraph {
        stacked_rgraph() = delete;
        using index_t          = cpu::stacked_rgraph::IndexStructure<index_traits_t>;
        using rgraph_config_t  = cpu::stacked_rgraph::RGraphConfig<index_traits_t>;
        using pruning_config_t = cpu::stacked_rgraph::PruningConfig<index_traits_t>;
    };

    /** @brief Namespace-scoped types for artea_graph
     *  (stacked_rgraph backbone with per-layer conv_graph refinement). */
    struct artea_graph {
        artea_graph() = delete;
        using index_t            = cpu::artea_graph::IndexStructure<index_traits_t>;
        using rgraph_config_t    = cpu::artea_graph::RGraphConfig<index_traits_t>;
        using propagate_config_t = cpu::artea_graph::PropagateConfig<index_traits_t>;
        using pruning_config_t   = cpu::artea_graph::PruningConfig<index_traits_t>;
    };

};  // struct IndexTraits

}   // namespace cpu
}   // namespace artea
