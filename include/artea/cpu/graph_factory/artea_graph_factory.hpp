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

namespace artea {
namespace cpu {

template <typename GraphFactoryTraitsT>
class ArteaGraphFactory :
    public HierarchicalGraphFactory<GraphFactoryTraitsT, ArteaGraphFactory<GraphFactoryTraitsT>>
{
    using dist_func_t = typename GraphFactoryTraitsT::dist_func_t;
    using flat_graph_t = typename GraphFactoryTraitsT::flat_graph_t;
    using conv_graph_factory_t = typename GraphFactoryTraitsT::conv_graph_factory_t;
    using hierarchical_graph_t = typename GraphFactoryTraitsT::hierarchical_graph_t;
    using vector_array_t = typename GraphFactoryTraitsT::vector_array_t;
    using hierarchical_vertices_builder_t = typename GraphFactoryTraitsT::hierarchical_vertices_builder_t;
    using hierarchical_edges_builder_t = typename GraphFactoryTraitsT::hierarchical_edges_builder_t;
    using layer_config_t = typename GraphFactoryTraitsT::layer_config_t;
    using greedy_vertices_builder_config_t = typename GraphFactoryTraitsT::greedy_vertices_builder_config_t;
    using pruning_config_t = typename GraphFactoryTraitsT::artea_graph::pruning_config_t;
    using propagate_config_t = typename GraphFactoryTraitsT::artea_graph::propagate_config_t;
    using vg_policy_t = typename GraphFactoryTraitsT::vg_policy_t;
    using eg_policy_t = typename GraphFactoryTraitsT::eg_policy_t;

public:

    template <vg_policy_t VGPolicy, eg_policy_t EGPolicy>
    static auto construct_graph_impl(
        const vector_array_t& base_vecs,
        layer_config_t bottom_layer_config,
        layer_config_t upper_layer_config,
        pruning_config_t bottom_pruning_config,
        pruning_config_t upper_pruning_config,
        propagate_config_t propagate_config,
        greedy_vertices_builder_config_t vertices_builder_config
    ) -> hierarchical_graph_t {
        dist_func_t dist_func(base_vecs.get_vec_dim());

        hierarchical_graph_t hierarchical_graph(
            base_vecs,
            bottom_layer_config,
            upper_layer_config,
            bottom_pruning_config,
            upper_pruning_config,
            propagate_config,
            vertices_builder_config
        );

        hierarchical_vertices_builder_t::template construct<VGPolicy>(
            dist_func,
            hierarchical_graph,
            vertices_builder_config
        );

        hierarchical_edges_builder_t::template construct<EGPolicy>(
            dist_func,
            hierarchical_graph
        );

        return hierarchical_graph;
    }
};

}   // namespace cpu
}   // namespace artea