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
 * @FilePath: /Artea/include/artea/cpu/graph_factory/hierarchical_search_graph_factory.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Date: 2026-03-13
 * @Description: Factory for converting HierarchicalGraph to HierarchicalSearchGraph.
 */

#pragma once

#include <memory>

namespace artea {
namespace cpu {

template <typename GraphFactoryTraitsT>
class HierarchicalSearchGraphFactory {

    using vertex_num_t = typename GraphFactoryTraitsT::vertex_num_t;
    using layer_id_t = typename GraphFactoryTraitsT::layer_id_t;
    using hierarchical_graph_t = typename GraphFactoryTraitsT::hierarchical_graph_t;
    using hierarchical_search_graph_t = typename GraphFactoryTraitsT::hierarchical_search_graph_t;
    using flat_search_graph_factory_t = typename GraphFactoryTraitsT::flat_search_graph_factory_t;

public:
    /**
     * @brief Factory method to convert HierarchicalGraph to HierarchicalSearchGraph.
     * @param hierarchical_graph The source hierarchical graph to convert from.
     * @param bl_extracted_nbr_size Fixed number of neighbors for bottom layer.
     * @param ul_extracted_nbr_size Fixed number of neighbors for upper layers.
     * @return A new HierarchicalSearchGraph instance.
     */
    static auto from_hierarchical_graph(
        const hierarchical_graph_t& hierarchical_graph,
        const vertex_num_t bl_extracted_nbr_size,
        const vertex_num_t ul_extracted_nbr_size
    ) -> hierarchical_search_graph_t {
        const auto& hier_vecs_manager = hierarchical_graph.get_hier_vecs_manager();
        const auto num_layers = hierarchical_graph.get_num_layers();

        hierarchical_search_graph_t hier_search_graph(
            hier_vecs_manager,
            bl_extracted_nbr_size,
            ul_extracted_nbr_size
        );

        hier_search_graph.resize(num_layers);

        // Convert bottom layer (layer_id = 0)
        const auto& bottom_flat_graph = hierarchical_graph.get_layer_graph(0);
        auto bottom_search_graph = flat_search_graph_factory_t::from_flat_graph(
            bottom_flat_graph,
            bl_extracted_nbr_size
        );
        hier_search_graph.set_layer_graph(0, std::move(bottom_search_graph));

        // Convert upper layers (layer_id > 0)
        for (layer_id_t layer_id = 1; layer_id < num_layers; ++layer_id) {
            const auto& upper_flat_graph = hierarchical_graph.get_layer_graph(layer_id);
            auto upper_search_graph = flat_search_graph_factory_t::from_flat_graph(
                upper_flat_graph,
                ul_extracted_nbr_size
            );
            hier_search_graph.set_layer_graph(layer_id, std::move(upper_search_graph));
        }

        // Copy inter-layer links and entry point
        hier_search_graph.get_inter_layer_links() = hierarchical_graph.get_inter_layer_links();
        hier_search_graph.set_entry_point(hierarchical_graph.get_entry_point());

        return hier_search_graph;
    }

    // TODO: add a from_index_file method like FlatSearchGraph

};  // class HierarchicalSearchGraphFactory

}   // namespace cpu
}   // namespace artea
