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
    using layer_config_t = typename GraphFactoryTraitsT::layer_config_t;
    using descent_config_t = typename GraphFactoryTraitsT::descent_config_t;
    using vg_policy_t = typename GraphFactoryTraitsT::vg_policy_t;
    using eg_policy_t = typename GraphFactoryTraitsT::eg_policy_t;

    static constexpr vertex_num_t min_num_vertex = 128;

public:

    /** @brief construct a new artea graph from vector array */
    template <vg_policy_t VGPolicy, eg_policy_t EGPolicy, typename... Args>
    static auto construct_graph_impl(
        const vector_array_t& base_vecs,
        const layer_config_t& bottom_layer_config,
        const layer_config_t& upper_layer_config,
        const descent_config_t& descent_config,
        Args&&... args
    ) -> hierarchical_graph_t {
        hierarchical_vecs_manager_t hier_vecs_manager(base_vecs);
        dist_func_t dist_func(base_vecs.get_vec_dim());
        if constexpr (EGPolicy == eg_policy_t::conv_graph_descent) {
            hierarchical_vertices_builder_t::template construct<VGPolicy>(
                base_vecs, dist_func, hier_vecs_manager, std::forward<Args>(args)...);
        }
    }
};

}   // namespace cpu
}   // namespace artea