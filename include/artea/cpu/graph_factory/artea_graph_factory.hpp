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
class ArteaGraphFactory :
    public GraphFactoryTraitsT::template hierarchical_graph_factory_t<ArteaGraphFactory<GraphFactoryTraitsT>>
{
    using vertex_num_t = typename GraphFactoryTraitsT::vertex_num_t;
    using vertex_id_t = typename GraphFactoryTraitsT::vertex_id_t;
    using layer_id_t = typename GraphFactoryTraitsT::layer_id_t;
    using layer_num_t = typename GraphFactoryTraitsT::layer_num_t;
    using ratio_t = typename GraphFactoryTraitsT::ratio_t;
    using distance_t = typename GraphFactoryTraitsT::distance_t;
    using dist_func_t = typename GraphFactoryTraitsT::dist_func_t;
    using flat_graph_t = typename GraphFactoryTraitsT::flat_graph_t;
    using conv_graph_factory_t = typename GraphFactoryTraitsT::conv_graph_factory_t;
    using hierarchical_graph_t = typename GraphFactoryTraitsT::hierarchical_graph_t;
    using vector_array_t = typename GraphFactoryTraitsT::vector_array_t;
    using vertex_subset_t = typename GraphFactoryTraitsT::vertex_subset_t;
    using lb_greedy_vg_t = typename GraphFactoryTraitsT::lb_greedy_vg_t;
    using centroid_computer_t = typename GraphFactoryTraitsT::centroid_computer_t;
    using bruteforce_router_t = typename GraphFactoryTraitsT::bruteforce_router_t;
    using layer_config_t = typename GraphFactoryTraitsT::layer_config_t;
    using greedy_vertices_builder_config_t = typename GraphFactoryTraitsT::greedy_vertices_builder_config_t;
    using pruning_config_t = typename GraphFactoryTraitsT::artea_graph::pruning_config_t;
    using propagate_config_t = typename GraphFactoryTraitsT::artea_graph::propagate_config_t;

    static constexpr vertex_num_t min_num_layer_vertex = GraphFactoryTraitsT::min_num_layer_vertex;

public:

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

        // ---- Construct hierarchical vertices (r-net greedy selection) ----
        auto& hier_vecs_manager = hierarchical_graph.get_hier_vecs_manager();
        auto& inter_layer_links = hierarchical_graph.get_inter_layer_links();

        layer_id_t current_layer_id = 0;
        const vector_array_t* current_layer_vecs = &base_vecs;
        distance_t current_radius = vertices_builder_config.min_radius() * vertices_builder_config.beta();

        while (true) {
            vertex_num_t max_result_size = static_cast<vertex_num_t>(
                current_layer_vecs->get_num_vecs() * vertices_builder_config.max_result_ratio()
            );

            lb_greedy_vg_t lb_greedy_vg(dist_func);
            vertex_subset_t next_layer_subset = lb_greedy_vg.generate(
                *current_layer_vecs,
                current_radius,
                max_result_size,
                vertices_builder_config.coverage_ratio(),
                vertices_builder_config.confidence(),
                vertices_builder_config.sampling_batch_size()
            );

            if (next_layer_subset.get_num_vecs() < min_num_layer_vertex) {
                break;
            }

            current_layer_id++;

            inter_layer_links.bottom_up_append(std::move(next_layer_subset.vec_ids));
            hier_vecs_manager.bottom_up_append(std::move(next_layer_subset.vecs_data));

            current_layer_vecs = &hier_vecs_manager.get_layer_vecs(current_layer_id);
            current_radius *= vertices_builder_config.beta();
        }

        // Find entry point: vertex closest to centroid in top layer
        {
            const layer_num_t num_layers = hier_vecs_manager.get_num_layers();
            if (num_layers == 0) {
                ARTEA_ERROR("Cannot find entry point: no layers in hierarchical graph");
            } else {
                const layer_id_t top_layer_id = num_layers - 1;
                const auto& top_layer_vecs = hier_vecs_manager.get_layer_vecs(top_layer_id);
                if (top_layer_vecs.get_num_vecs() == 0) {
                    ARTEA_ERROR("Cannot find entry point: top layer is empty");
                } else {
                    auto centroid = centroid_computer_t::compute(top_layer_vecs);
                    bruteforce_router_t bf_router(top_layer_vecs, dist_func, 1);
                    bf_router.initialize();
                    auto nearest_vertices = bf_router.query(centroid.data());
                    if (nearest_vertices.empty()) {
                        ARTEA_ERROR("Cannot find entry point: bruteforce router returned empty result");
                    } else {
                        hierarchical_graph.set_entry_point(nearest_vertices[0].get_id());
                    }
                }
            }
        }

        // ---- Construct edges (convergent graph descent) ----
        const auto num_layers = hierarchical_graph.get_num_layers();
        hierarchical_graph.resize(num_layers);

        conv_graph_factory_t conv_factory;

        auto bottom_graph = conv_factory.construct_graph(
            base_vecs,
            hierarchical_graph.bottom_layer_config(),
            hierarchical_graph.bottom_pruning_config(),
            hierarchical_graph.propagate_config()
        );
        hierarchical_graph.set_layer_graph(0, std::move(bottom_graph));

        for (layer_id_t layer_id = 1; layer_id < num_layers; ++layer_id) {
            const auto& layer_vecs = hier_vecs_manager.get_layer_vecs(layer_id);
            auto upper_graph = conv_factory.construct_graph(
                layer_vecs,
                hierarchical_graph.upper_layer_config(),
                hierarchical_graph.upper_pruning_config(),
                hierarchical_graph.propagate_config()
            );
            hierarchical_graph.set_layer_graph(layer_id, std::move(upper_graph));
        }

        return hierarchical_graph;
    }
};

}   // namespace cpu
}   // namespace artea