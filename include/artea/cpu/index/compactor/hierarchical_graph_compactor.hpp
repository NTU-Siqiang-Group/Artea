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
 * @FilePath: /Artea/include/artea/cpu/index/compactor/hierarchical_graph_compactor.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Compactor: dynamic::HierarchicalGraph <-> compact::HierarchicalGraph.
 */

#pragma once

#include <memory>
#include <artea/common/logger.hpp>

namespace artea {
namespace cpu {

/**
 * @brief Converts between dynamic::HierarchicalGraph and
 *        compact::HierarchicalGraph by compacting / expanding each layer
 *        via InternalGraphCompactor.
 *
 * @tparam IndexTraitsT The index traits type.
 */
template <typename IndexTraitsT>
class HierarchicalGraphCompactor {

    using vertex_num_t = typename IndexTraitsT::vertex_num_t;
    using layer_num_t  = typename IndexTraitsT::layer_num_t;
    using layer_id_t   = typename IndexTraitsT::layer_id_t;
    using compact      = typename IndexTraitsT::compact;
    using dynamic      = typename IndexTraitsT::dynamic;
    using internal_graph_compactor_t = typename IndexTraitsT::internal_graph_compactor_t;

public:
    /**
     * @brief Compact a dynamic::HierarchicalGraph into a compact::HierarchicalGraph.
     *
     * Each layer is compacted independently via InternalGraphCompactor::compact_graph.
     *
     * @param src                The source dynamic hierarchical graph.
     * @param extracted_nbr_size Fixed neighbor slots per vertex per layer.
     * @return A new compact::HierarchicalGraph.
     */
    static auto compact_graph(
        const typename dynamic::hierarchical_graph_t& src,
        const vertex_num_t extracted_nbr_size
    ) -> typename compact::hierarchical_graph_t {
        const layer_num_t num_layers = src.get_num_layers();
        typename compact::hierarchical_graph_t result(num_layers);

        for (layer_id_t l = 0; l < num_layers; ++l) {
            const auto& src_layer = src.get_layer_graph(l);
            auto compact_layer = std::make_unique<typename compact::internal_graph_t>(
                internal_graph_compactor_t::compact_graph(src_layer, extracted_nbr_size)
            );
            result.set_layer_graph(l, std::move(compact_layer));
        }

        return result;
    }

};  // class HierarchicalGraphCompactor

}   // namespace cpu
}   // namespace artea
