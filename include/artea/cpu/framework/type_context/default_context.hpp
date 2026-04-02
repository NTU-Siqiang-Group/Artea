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

// Type definitions using EUCLIDEAN, LOCKED_BUFFER_WITH_MUTEX
using vec_num_t = uint32_t;
using vec_ele_t = float;

using base_traits_t = BaseTraits<vec_num_t, vec_ele_t>;
using computer_traits_t = ComputerTraits<base_traits_t, DistanceMetricsT::EUCLIDEAN>;
using buffer_traits_t = BufferTraits<base_traits_t, BufferPolicyT::LOCKED_BUFFER_WITH_MUTEX, 32>;
using index_traits_t = IndexTraits<base_traits_t>;
using vertex_generator_traits_t = VertexGeneratorTraits<computer_traits_t, index_traits_t>;
using router_traits_t = RouterTraits<computer_traits_t, index_traits_t, false>;
using edge_generator_traits_t = EdgeGeneratorTraits<computer_traits_t, buffer_traits_t, index_traits_t, router_traits_t>;
using graph_factory_traits_t = GraphFactoryTraits<
    vertex_generator_traits_t,
    edge_generator_traits_t,
    router_traits_t
>;

// Base types from BaseTraits
using vec_dim_t = typename base_traits_t::vec_dim_t;
using vertex_num_t = typename base_traits_t::vertex_num_t;
using vertex_id_t = typename base_traits_t::vertex_id_t;
using vec_id_t = typename base_traits_t::vec_id_t;
using vec_ele_t = typename base_traits_t::vec_ele_t;
using distance_t = typename base_traits_t::distance_t;
using ratio_t = typename base_traits_t::ratio_t;
using cluster_num_t = typename base_traits_t::cluster_num_t;
using cluster_id_t = typename base_traits_t::cluster_id_t;
using part_num_t = typename base_traits_t::part_num_t;
using part_id_t = typename base_traits_t::part_id_t;
using batch_id_t = typename base_traits_t::batch_id_t;
using layer_num_t = typename base_traits_t::layer_num_t;
using layer_id_t = typename base_traits_t::layer_id_t;
using hash_num_t = typename base_traits_t::hash_num_t;
using iter_t = typename base_traits_t::iter_t;
using nbr_t = typename base_traits_t::nbr_t;
using nbr_arr_t = typename base_traits_t::nbr_arr_t;
using nbr_comp_t = typename base_traits_t::nbr_comp_t;
using strict_nbr_comp_t = typename base_traits_t::strict_nbr_comp_t;
using nbr_id_comp_t = typename base_traits_t::nbr_id_comp_t;
using nbr_dist_comp_t = typename base_traits_t::nbr_dist_comp_t;
using word_aligned_bitmap_t = typename base_traits_t::word_aligned_bitmap_t;
using thread_local_bitmap_t = typename base_traits_t::thread_local_bitmap_t;
using version_tag_table_t = typename base_traits_t::version_tag_table_t;
using csr_vids_t = typename base_traits_t::csr_vids_t;
using vector_t = typename base_traits_t::vector_t;
using vector_array_t = typename base_traits_t::vector_array_t;
using idlist_array_t = typename base_traits_t::idlist_array_t;
using vector_dataset_t = typename base_traits_t::vector_dataset_t;
using base_vecs_t = typename base_traits_t::base_vecs_t;
using query_vecs_t = typename base_traits_t::query_vecs_t;
using ground_truth_t = typename base_traits_t::ground_truth_t;
using vector_sampler_t = typename base_traits_t::vector_sampler_t;
using nbr_arr_checker_t = typename base_traits_t::nbr_arr_checker_t;
using random_seq_t = typename base_traits_t::random_seq_t;
using random_seq_nr_t = typename base_traits_t::random_seq_nr_t;
using vertex_subset_t = typename base_traits_t::vertex_subset_t;
using pruning_condition_t = typename base_traits_t::pruning_condition_t;
using centroid_computer_t = typename base_traits_t::centroid_computer_t;

// Computer types from ComputerTraits
using distance_metrics_t = typename computer_traits_t::distance_metrics_t;
using dist_func_t = typename computer_traits_t::dist_func_t;
using simdu1_dist_t = typename computer_traits_t::simdu1_dist_t;
using simdu2_dist_t = typename computer_traits_t::simdu2_dist_t;
using simdu4_dist_t = typename computer_traits_t::simdu4_dist_t;
using fma_func_t = typename computer_traits_t::fma_func_t;
using simdu1_fma_t = typename computer_traits_t::simdu1_fma_t;
using simdu2_fma_t = typename computer_traits_t::simdu2_fma_t;
using simdu4_fma_t = typename computer_traits_t::simdu4_fma_t;
using linear_func_t = typename computer_traits_t::linear_func_t;
using simdu1_linear_t = typename computer_traits_t::simdu1_linear_t;
using simdu2_linear_t = typename computer_traits_t::simdu2_linear_t;
using simdu4_linear_t = typename computer_traits_t::simdu4_linear_t;
using recall_estimator_t = typename computer_traits_t::recall_estimator_t;
using distance_prober_t = typename computer_traits_t::distance_prober_t;

// Buffer types from BufferTraits
using buffer_policy_t = typename buffer_traits_t::buffer_policy_t;
using log_buffer_t = typename buffer_traits_t::log_buffer_t;
using log_container_t = typename buffer_traits_t::log_container_t;
using log_table_t = typename buffer_traits_t::log_table_t;

// Config types from BaseTraits
using layer_config_t = typename base_traits_t::layer_config_t;
using greedy_vertices_builder_config_t = typename base_traits_t::greedy_vertices_builder_config_t;
using random_vertices_builder_config_t = typename base_traits_t::random_vertices_builder_config_t;
using vertices_builder_config_t = typename base_traits_t::vertices_builder_config_t;

// Index types from IndexTraits
using flat_search_graph_t = typename index_traits_t::flat_search_graph_t;
using hierarchical_search_graph_t = typename index_traits_t::hierarchical_search_graph_t;
using inter_layer_links_t = typename index_traits_t::inter_layer_links_t;
using hierarchical_vecs_manager_t = typename index_traits_t::hierarchical_vecs_manager_t;
using search_graph_converter_t = typename index_traits_t::search_graph_converter_t;
using flat_graph_file_manager_t = typename index_traits_t::flat_graph_file_manager_t;
using hierarchical_graph_file_manager_t = typename index_traits_t::hierarchical_graph_file_manager_t;
using index_size_calculator_t = typename index_traits_t::index_size_calculator_t;
using radius_prober_t = typename index_traits_t::radius_prober_t;

// Edge generator types from EdgeGeneratorTraits
using triangle_updater_t = typename edge_generator_traits_t::template triangle_updater_t<typename index_traits_t::conv_graph::index_t>;
using reverse_updater_t = typename edge_generator_traits_t::template reverse_updater_t<typename index_traits_t::conv_graph::index_t>;
using random_updater_t = typename edge_generator_traits_t::template random_updater_t<typename index_traits_t::conv_graph::index_t>;
using routing_updater_t = typename edge_generator_traits_t::template routing_updater_t<typename index_traits_t::conv_graph::index_t>;
using truncate_updater_t = typename edge_generator_traits_t::template truncate_updater_t<typename index_traits_t::conv_graph::index_t>;
using random_eg_t = typename edge_generator_traits_t::random_eg_t;
using ivf_partitions_t = typename edge_generator_traits_t::ivf_partitions_t;
using ivf_construct_policy_t = typename edge_generator_traits_t::ivf_construct_policy_t;

// Vertex generator types from VertexGeneratorTraits
using approx_rnet_t = typename vertex_generator_traits_t::approx_rnet_t;
using ortho_lsh_generator_t = typename vertex_generator_traits_t::ortho_lsh_generator_t;
using pstable_lsh_generator_t = typename vertex_generator_traits_t::pstable_lsh_generator_t;
using lsh_table_t = typename vertex_generator_traits_t::lsh_table_t;
using lb_greedy_vg_t = typename vertex_generator_traits_t::lb_greedy_vg_t;
using mb_greedy_vg_t = typename vertex_generator_traits_t::mb_greedy_vg_t;
using random_vg_t = typename vertex_generator_traits_t::random_vg_t;
using graph_mis_vg_t = typename vertex_generator_traits_t::graph_mis_vg_t;

// Router types from RouterTraits
using candidate_entry_t = typename router_traits_t::candidate_entry_t;
using result_entry_t = typename router_traits_t::result_entry_t;
using knn_results_t = typename router_traits_t::knn_results_t;
using std_candidate_queue_t = typename router_traits_t::std_candidate_queue_t;
using linear_candidate_queue_t = typename router_traits_t::linear_candidate_queue_t;
using fh_candidate_queue_t = typename router_traits_t::fh_candidate_queue_t;
using candidate_queue_t = typename router_traits_t::candidate_queue_t;
using visited_table_pool_t = typename router_traits_t::visited_table_pool_t;
using bruteforce_router_t = typename router_traits_t::bruteforce_router_t;

// alias for routers with customizable GraphModeT
using graph_mode_t = typename router_traits_t::graph_mode_t;

template <GraphModeT Mode = GraphModeT::search_mode>
using monolayer_graph_router_t = typename router_traits_t::template monolayer_graph_router_t<Mode>;

template <GraphModeT Mode = GraphModeT::search_mode>
using hierarchical_graph_router_t = typename router_traits_t::template hierarchical_graph_router_t<Mode>;

// Utility types
using index_register_util_t = IndexRegisterUtil;

// Propagate engine from EdgeGeneratorTraits
using propagate_engine_ss_t = typename edge_generator_traits_t::template propagate_engine_t<typename index_traits_t::conv_graph::index_t, true>;
using propagate_engine_noss_t = typename edge_generator_traits_t::template propagate_engine_t<typename index_traits_t::conv_graph::index_t, false>;
// Currently, NO SELECTIVE SCHEDULING is faster
using propagate_engine_t = propagate_engine_noss_t;

// Namespace-scoped types from GraphFactoryTraits (index_t, factory_t, config types)
namespace conv_graph {
    using index_t = typename graph_factory_traits_t::conv_graph::index_t;
    using factory_t = typename graph_factory_traits_t::conv_graph::factory_t;
    using propagate_config_t = typename graph_factory_traits_t::conv_graph::propagate_config_t;
    using pruning_config_t = typename graph_factory_traits_t::conv_graph::pruning_config_t;
}

namespace knn_graph {
    using index_t = typename graph_factory_traits_t::knn_graph::index_t;
    using factory_t = typename graph_factory_traits_t::knn_graph::factory_t;
    using propagate_config_t = typename graph_factory_traits_t::knn_graph::propagate_config_t;
    using pruning_config_t = typename graph_factory_traits_t::knn_graph::pruning_config_t;
}

namespace symmetric_knn_graph {
    using index_t = typename graph_factory_traits_t::symmetric_knn_graph::index_t;
    using factory_t = typename graph_factory_traits_t::symmetric_knn_graph::factory_t;
    using propagate_config_t = typename graph_factory_traits_t::symmetric_knn_graph::propagate_config_t;
    using pruning_config_t = typename graph_factory_traits_t::symmetric_knn_graph::pruning_config_t;
}

namespace artea_graph {
    using index_t = typename graph_factory_traits_t::artea_graph::index_t;
    using factory_t = typename graph_factory_traits_t::artea_graph::factory_t;
    using propagate_config_t = typename graph_factory_traits_t::artea_graph::propagate_config_t;
    using pruning_config_t = typename graph_factory_traits_t::artea_graph::pruning_config_t;
}

}   // namespace cpu
}   // namespace artea