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
    public GraphFactoryTraitsT::template hierarchical_graph_factory_t<ArteaGraphFactory<GraphFactoryTraitsT>>
{
    using vertex_num_t = typename GraphFactoryTraitsT::vertex_num_t;
    using vertex_id_t = typename GraphFactoryTraitsT::vertex_id_t;
    using layer_id_t = typename GraphFactoryTraitsT::layer_id_t;
    using vec_ele_t = typename GraphFactoryTraitsT::vec_ele_t;
    using iter_t = typename GraphFactoryTraitsT::iter_t;
    using ratio_t = typename GraphFactoryTraitsT::ratio_t;
    using distance_t = typename GraphFactoryTraitsT::distance_t;
    using dist_func_t = typename GraphFactoryTraitsT::dist_func_t;
    using flat_graph_t = typename GraphFactoryTraitsT::flat_graph_t;
    using conv_graph_t = typename GraphFactoryTraitsT::conv_graph_t;
    using conv_graph_factory_t = typename GraphFactoryTraitsT::conv_graph_factory_t;
    using hierarchical_graph_t = typename GraphFactoryTraitsT::hierarchical_graph_t;
    using vector_dataset_t = typename GraphFactoryTraitsT::vector_dataset_t;
    using vector_array_t = typename GraphFactoryTraitsT::vector_array_t;
    using vertex_subset_t = typename GraphFactoryTraitsT::vertex_subset_t;
    using random_vg_t = typename GraphFactoryTraitsT::random_vg_t;
    using lb_greedy_vg_t = typename GraphFactoryTraitsT::lb_greedy_vg_t;
    using hierarchical_vecs_manager_t = typename GraphFactoryTraitsT::hierarchical_vecs_manager_t;
    using hierarchical_vertices_builder_t = typename GraphFactoryTraitsT::hierarchical_vertices_builder_t;
    using hierarchical_edges_builder_t = typename GraphFactoryTraitsT::hierarchical_edges_builder_t;
    using layer_config_t = typename GraphFactoryTraitsT::layer_config_t;
    using edges_builder_config_t = typename GraphFactoryTraitsT::artea_graph::edges_builder_config_t;
    using greedy_vertices_builder_config_t = typename GraphFactoryTraitsT::greedy_vertices_builder_config_t;
    using random_vertices_builder_config_t = typename GraphFactoryTraitsT::random_vertices_builder_config_t;
    using vg_policy_t = typename GraphFactoryTraitsT::vg_policy_t;
    using eg_policy_t = typename GraphFactoryTraitsT::eg_policy_t;

public:

    /** @brief construct a new artea graph from vector array with greedy vertices builder */
    template <vg_policy_t VGPolicy, eg_policy_t EGPolicy>
    static auto construct_graph_impl(
        const vector_array_t& base_vecs,
        layer_config_t bottom_layer_config,
        layer_config_t upper_layer_config,
        edges_builder_config_t bottom_edges_builder_config,
        edges_builder_config_t upper_edges_builder_config,
        greedy_vertices_builder_config_t vertices_builder_config
    ) -> hierarchical_graph_t requires (VGPolicy == vg_policy_t::rnet_selection) {
        dist_func_t dist_func(base_vecs.get_vec_dim());

        // Create hierarchical graph
        hierarchical_graph_t hierarchical_graph(
            base_vecs,
            bottom_layer_config,
            upper_layer_config,
            bottom_edges_builder_config,
            upper_edges_builder_config,
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

    /** @brief construct a new artea graph from vector array with random vertices builder */
    template <vg_policy_t VGPolicy, eg_policy_t EGPolicy>
    static auto construct_graph_impl(
        const vector_array_t& base_vecs,
        layer_config_t bottom_layer_config,
        layer_config_t upper_layer_config,
        edges_builder_config_t bottom_edges_builder_config,
        edges_builder_config_t upper_edges_builder_config,
        random_vertices_builder_config_t vertices_builder_config
    ) -> hierarchical_graph_t requires (VGPolicy == vg_policy_t::random_selection) {
        dist_func_t dist_func(base_vecs.get_vec_dim());

        // Create hierarchical graph
        hierarchical_graph_t hierarchical_graph(
            base_vecs,
            bottom_layer_config,
            upper_layer_config,
            bottom_edges_builder_config,
            upper_edges_builder_config,
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