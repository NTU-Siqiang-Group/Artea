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

template <typename GraphFactoryTraitsT>
class ArteaGraphFactory :
    public GraphFactoryTraitsT::template hierarchical_graph_factory_t<ArteaGraphFactory<GraphFactoryTraitsT>>,
{
    using vertex_num_t = typename GraphFactoryTraitsT::vertex_num_t;
    using vertex_id_t = typename GraphFactoryTraitsT::vertex_id_t;
    using vec_ele_t = typename GraphFactoryTraitsT::vec_ele_t;
    using iter_t = typename GraphFactoryTraitsT::iter_t;
    using ratio_t = typename GraphFactoryTraitsT::ratio_t;
    using dist_func_t = typename GraphFactoryTraitsT::dist_func_t;
    using flat_graph_t = typename GraphFactoryTraitsT::flat_graph_t;
    using conv_graph_t = typename GraphFactoryTraitsT::conv_graph_t;
    using conv_graph_factory_t = typename GraphFactoryTraitsT::conv_graph_factory_t;
    using hierarchical_graph_t = typename GraphFactoryTraitsT::hierarchical_graph_t;
    using vector_dataset_t = typename GraphFactoryTraitsT::vector_dataset_t;
    using vector_array_t = typename GraphFactoryTraitsT::vector_array_t;

    // vertex generators type
    using lb_greedy_vg_t = typename GraphFactoryTraitsT::lb_greedy_vg_t;
    using random_vg_t = typename RandomVG<vertex_generator_traits_t>;

    /** @brief construct a new artea graph from vector array */
    static auto construct_graph_impl(
        const vector_array_t& base_vecs,
        const vertex_num_t bl_max_nbr_size,
        const vertex_num_t ul_max_nbr_size,
        const vertex_num_t bl_reserved_nbr_size,
        const vertex_num_t ul_reserved_nbr_size,
        const distance_t min_radius,
        const ratio_t beta_factor,
        const ratio_t scale_coeffs,
        const ratio_t shifted_coeffs,
        const iter_t num_outer_iters,   // recommend param: 4
        const iter_t num_inner_iters    // recommend param: 14
    ) -> hierarchical_graph_t {

    }

    // auto construct_hierarchical_vertex(

    // ) ->
};