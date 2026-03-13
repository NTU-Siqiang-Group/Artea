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

#include <artea/cpu/graph_factory/artea_graph_factory.hpp>

namespace artea {
namespace cpu {

template <typename GraphFactoryTraitsT>
class HierarchicalVerticesBuilder {
    using vertex_num_t = typename GraphFactoryTraitsT::vertex_num_t;
    using vertex_id_t = typename GraphFactoryTraitsT::vertex_id_t;
    using layer_id_t = typename GraphFactoryTraitsT::layer_id_t;
    using ratio_t = typename GraphFactoryTraitsT::ratio_t;
    using distance_t = typename GraphFactoryTraitsT::distance_t;
    using dist_func_t = typename GraphFactoryTraitsT::dist_func_t;
    using vector_array_t = typename GraphFactoryTraitsT::vector_array_t;
    using vertex_subset_t = typename GraphFactoryTraitsT::vertex_subset_t;
    using random_vg_t = typename GraphFactoryTraitsT::random_vg_t;
    using lb_greedy_vg_t = typename GraphFactoryTraitsT::lb_greedy_vg_t;
    using hierarchical_vecs_manager_t = typename GraphFactoryTraitsT::hierarchical_vecs_manager_t;
    using hierarchical_graph_t = typename GraphFactoryTraitsT::hierarchical_graph_t;
    using vg_policy_t = typename GraphFactoryTraitsT::vg_policy_t;

    static constexpr vertex_num_t min_num_vertex = 128;

public:
    // Specialization for rnet_selection
    template <vg_policy_t VGPolicy>
    static auto construct(
        const dist_func_t& dist_func,
        hierarchical_graph_t& hierarchical_graph,
        const distance_t rnet_radius,
        const ratio_t beta_sq,
        const ratio_t coverage_ratio,
        const ratio_t confidence,
        const vertex_num_t max_result_size,
        const vertex_num_t sampling_batch_size,
        const bool is_shuffle
    ) -> void requires (VGPolicy == vg_policy_t::rnet_selection) {
        // Get base_vecs from hierarchical_graph
        const auto& base_vecs = hierarchical_graph.get_base_vecs();
        const vertex_num_t num_vertices = static_cast<vertex_num_t>(base_vecs.get_num_vecs());

        // Get references to hier_vecs_manager and inter_layer_links
        auto& hier_vecs_manager = hierarchical_graph.get_hier_vecs_manager();
        auto& inter_layer_links = hierarchical_graph.get_inter_layer_links();

        // Layer 0 is the base_vecs (full dataset), already in hier_vecs_manager by construction
        // Start building from Layer 1 with radius = rnet_radius * beta_sq

        // Build upper layers iteratively
        layer_id_t current_layer_id = 0;
        const vector_array_t* current_layer_vecs = &base_vecs;
        distance_t current_radius = rnet_radius * beta_sq;  // Layer 1 starts with min_radius * beta_sq

        while (true) {
            // Generate next layer from current layer with scaled radius
            vertex_subset_t next_layer_subset = _construct_hier_vertex_lb_greedy(
                *current_layer_vecs,
                dist_func,
                current_radius,
                beta_sq,
                coverage_ratio,
                confidence,
                max_result_size,
                sampling_batch_size,
                is_shuffle
            );

            // Check termination condition
            if (next_layer_subset.get_num_vecs() < min_num_vertex) {
                break;
            }

            // Update layer_id for the new layer
            current_layer_id++;

            // Add inter-layer links and vector data (move semantics)
            inter_layer_links.bottom_up_append(std::move(next_layer_subset.vec_ids));
            hier_vecs_manager.bottom_up_append(std::move(next_layer_subset.vecs_data));

            // Update for next iteration
            current_layer_vecs = &hier_vecs_manager.get_layer_vecs(current_layer_id);
            current_radius *= beta_sq;  // Scale radius for next layer
        }
    }

    // Specialization for random_selection
    template <vg_policy_t VGPolicy>
    static auto construct(
        const dist_func_t& dist_func,
        hierarchical_graph_t& hierarchical_graph,
        const ratio_t result_ratio
    ) -> void requires (VGPolicy == vg_policy_t::random_selection) {
        // Get base_vecs from hierarchical_graph
        const auto& base_vecs = hierarchical_graph.get_base_vecs();
        const vertex_num_t num_vertices = static_cast<vertex_num_t>(base_vecs.get_num_vecs());

        // Get references to hier_vecs_manager and inter_layer_links
        auto& hier_vecs_manager = hierarchical_graph.get_hier_vecs_manager();
        auto& inter_layer_links = hierarchical_graph.get_inter_layer_links();

        // Layer 0 is the base_vecs (full dataset), already in hier_vecs_manager by construction

        // Build upper layers iteratively
        layer_id_t current_layer_id = 0;
        const vector_array_t* current_layer_vecs = &base_vecs;

        while (true) {
            // Calculate result size based on current layer size and ratio
            vertex_num_t current_result_size = static_cast<vertex_num_t>(
                current_layer_vecs->get_num_vecs() * result_ratio
            );

            // Generate next layer from current layer
            vertex_subset_t next_layer_subset = _construct_hier_vertex_random(
                *current_layer_vecs,
                dist_func,
                current_result_size
            );

            // Check termination condition
            if (next_layer_subset.get_num_vecs() < min_num_vertex) {
                break;
            }

            // Update layer_id for the new layer
            current_layer_id++;

            // Add inter-layer links and vector data (move semantics)
            inter_layer_links.bottom_up_append(std::move(next_layer_subset.vec_ids));
            hier_vecs_manager.bottom_up_append(std::move(next_layer_subset.vecs_data));

            // Update for next iteration
            current_layer_vecs = &hier_vecs_manager.get_layer_vecs(current_layer_id);
        }
    }

private:
    static auto _construct_hier_vertex_lb_greedy(
        const vector_array_t& layer_vecs,
        const dist_func_t& dist_func,
        const distance_t rnet_radius,
        const ratio_t beta_sq,
        const ratio_t coverage_ratio,
        const ratio_t confidence,
        const vertex_num_t max_result_size,
        const vertex_num_t sampling_batch_size,
        const bool is_shuffle
    ) -> vertex_subset_t {
        lb_greedy_vg_t lb_greedy_vg(dist_func);

        // Generate approximate r-net for this layer with the given radius
        return lb_greedy_vg.generate(
            layer_vecs,
            rnet_radius,
            max_result_size,
            coverage_ratio,
            confidence,
            sampling_batch_size,
            is_shuffle
        );
    }

    static auto _construct_hier_vertex_random(
        const vector_array_t& layer_vecs,
        const dist_func_t& dist_func,
        const vertex_num_t result_size
    ) -> vertex_subset_t {
        random_vg_t random_vg;
        return random_vg.generate(layer_vecs, result_size);
    }
};

}  // namespace cpu
}  // namespace artea