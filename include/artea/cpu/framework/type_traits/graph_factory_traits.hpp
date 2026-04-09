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
namespace knn_graph {
    template <typename GraphFactoryTraitsT> class IndexFactory;
}
namespace symmetric_knn_graph {
    template <typename GraphFactoryTraitsT> class IndexFactory;
}
namespace stacked_rgraph {
    template <typename GraphFactoryTraitsT> class IndexFactory;
}

template <
    typename VertexGeneratorTraitsT,
    typename RefinerTraitsT,
    typename RouterTraitsT
>
struct GraphFactoryTraits :
    public VertexGeneratorTraitsT,
    public RefinerTraitsT
{
    using graph_factory_traits_t = GraphFactoryTraits<VertexGeneratorTraitsT, RefinerTraitsT, RouterTraitsT>;

    /** @brief Namespace-scoped factory types for conv_graph, extending IndexTraits::conv_graph. */
    struct conv_graph : RefinerTraitsT::conv_graph {
        conv_graph() = delete;
        using factory_t = cpu::conv_graph::IndexFactory<graph_factory_traits_t>;
    };

    /** @brief Namespace-scoped factory types for knn_graph, extending IndexTraits::knn_graph. */
    struct knn_graph : RefinerTraitsT::knn_graph {
        knn_graph() = delete;
        using factory_t = cpu::knn_graph::IndexFactory<graph_factory_traits_t>;
    };

    /** @brief Namespace-scoped factory types for symmetric_knn_graph, extending IndexTraits::symmetric_knn_graph. */
    struct symmetric_knn_graph : RefinerTraitsT::symmetric_knn_graph {
        symmetric_knn_graph() = delete;
        using factory_t = cpu::symmetric_knn_graph::IndexFactory<graph_factory_traits_t>;
    };

    /** @brief Namespace-scoped factory types for artea_graph, extending IndexTraits::artea_graph. */
    struct artea_graph : RefinerTraitsT::artea_graph {
        artea_graph() = delete;
        using factory_t = cpu::artea_graph::IndexFactory<graph_factory_traits_t>;
    };

    /** @brief Namespace-scoped factory types for stacked_rgraph, extending IndexTraits::stacked_rgraph. */
    struct stacked_rgraph : RefinerTraitsT::stacked_rgraph {
        stacked_rgraph() = delete;
        using factory_t = cpu::stacked_rgraph::IndexFactory<graph_factory_traits_t>;
    };

};  // struct GraphFactoryTraits

}   // namespace cpu
}   // namespace artea
