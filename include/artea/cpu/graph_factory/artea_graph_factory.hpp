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

enum class VGPolicyT {
    ramdom_selection,
    rnet_selection
};

template <typename GraphFactoryTraitsT>
class ArteaGraphFactory :
    public GraphFactoryTraitsT::template hierarchical_graph_factory_t<ArteaGraphFactory<GraphFactoryTraitsT>>,
{
    using vertex_num_t = typename GraphFactoryTraitsT::vertex_num_t;
    using vertex_id_t = typename GraphFactoryTraitsT::vertex_id_t;
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
    using random_vg_t = typename GraphFactoryTraitsT::random_vg_t;
    using lb_greedy_vg_t = typename GraphFactoryTraitsT::lb_greedy_vg_t;
    using hierarchical_vecs_manager_t = typename GraphFactoryTraitsT::hierarchical_vecs_manager_t;
    using layer_config_t = typename GraphFactoryTraitsT::layer_config_t;
    using descent_config_t = typename GraphFactoryTraitsT::descent_config_t;

public:

    /** @brief construct a new artea graph from vector array */
    static auto construct_graph_impl(
        const vector_array_t& base_vecs,
        const layer_config_t& bottom_layer_config,
        const layer_config_t& upper_layer_config,
        const descent_config_t& descent_config,
        const ratio_t beta_factor,
        const distance_t min_radius
    ) -> hierarchical_graph_t {

    }

private:
    template <VGPolicyT VGPolicy>
    auto _construct_hier_vertex(
        const vector_array_t& base_vecs,
        const dist_func_t& dist_func,
        hierarchical_vecs_manager_t& hier_vecs_manager,
        // optional args
    ) -> void {
        const vertex_num_t num_vertices = static_cast<vertex_num_t>(base_vecs.get_num_vecs());
        if constexpr (VGPolicy == VGPolicyT::ramdom_selection) {

        }
        else if constexpr (VGPolicy == VGPolicyT::rnet_selection) {

        }
    }

    auto _construct_hier_vertex_lb_greedy(
        const vector_array_t& base_vecs,
        const dist_func_t& dist_func,
        hierarchical_vecs_manager_t& hier_vecs_manager
    ) -> void {
        lb_greedy_vg_t lb_greedy_vg(dist_func);
        lb_greedy_vg.generate()
    }
};