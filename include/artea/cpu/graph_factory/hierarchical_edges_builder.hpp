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

#pragma once

#include <artea/common/logger.hpp>

namespace artea {
namespace cpu {

template <typename GraphFactoryTraitsT>
class HierarchicalEdgesBuilder {

    using vertex_num_t = typename GraphFactoryTraitsT::vertex_num_t;
    using vertex_id_t = typename GraphFactoryTraitsT::vertex_id_t;
    using layer_id_t = typename GraphFactoryTraitsT::layer_id_t;
    using layer_num_t = typename GraphFactoryTraitsT::layer_num_t;
    using ratio_t = typename GraphFactoryTraitsT::ratio_t;
    using distance_t = typename GraphFactoryTraitsT::distance_t;
    using dist_func_t = typename GraphFactoryTraitsT::dist_func_t;
    using vector_array_t = typename GraphFactoryTraitsT::vector_array_t;
    using hierarchical_graph_t = typename GraphFactoryTraitsT::hierarchical_graph_t;
    using conv_graph_factory_t = typename GraphFactoryTraitsT::conv_graph_factory_t;
    using eg_policy_t = typename GraphFactoryTraitsT::eg_policy_t;

public:
    // Specialization for conv_graph_descent
    template <eg_policy_t EGPolicy>
    static auto construct(
        const dist_func_t& dist_func,
        hierarchical_graph_t& hierarchical_graph
    ) -> void requires (EGPolicy == eg_policy_t::conv_graph_descent) {
        const auto& base_vecs = hierarchical_graph.get_base_vecs();
        const auto num_layers = hierarchical_graph.get_num_layers();

        // Resize layer_graphs to match the number of layers after vertices construction
        hierarchical_graph.resize(num_layers);

        conv_graph_factory_t conv_factory;

        // Construct bottom layer (layer_id = 0)
        auto bottom_graph = conv_factory.construct_graph(
            base_vecs,
            hierarchical_graph.bottom_layer_config(),
            hierarchical_graph.bottom_edges_builder_config()
        );
        hierarchical_graph.set_layer_graph(0, std::move(bottom_graph));

        // Construct upper layers (layer_id > 0)
        for (layer_id_t layer_id = 1; layer_id < num_layers; ++layer_id) {
            const auto& layer_vecs = hierarchical_graph.get_hier_vecs_manager().get_layer_vecs(layer_id);
            auto upper_graph = conv_factory.construct_graph(
                layer_vecs,
                hierarchical_graph.upper_layer_config(),
                hierarchical_graph.upper_edges_builder_config()
            );
            hierarchical_graph.set_layer_graph(layer_id, std::move(upper_graph));
        }
    }

    // Specialization for speculative_conv_graph_descent
    template <eg_policy_t EGPolicy>
    static auto construct(
        const dist_func_t& dist_func,
        hierarchical_graph_t& hierarchical_graph
    ) -> void requires (EGPolicy == eg_policy_t::speculative_conv_graph_descent) {
        logger.error("HierarchicalEdgesBuilder::construct<speculative_conv_graph_descent> not implemented yet");
    }

};  // class HierarchicalEdgesBuilder

}   // namespace cpu
}   // namespace artea