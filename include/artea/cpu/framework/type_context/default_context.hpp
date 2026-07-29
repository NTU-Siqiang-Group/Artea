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
 * @FilePath: /Artea/include/artea/cpu/framework/type_context/default_context.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Public artea type aliases.
 *
 *   Two compile-time axes are explicit at every use site:
 *     - Metric  : the distance metric (DistanceMetricsT)
 *     - Dim : the SIMD-padded vector dimension
 *   ONLY the types that depend on ComputerTraits are parameterized by <Metric, Dim>;
 *   everything that derives purely from BaseTraits / BufferTraits / IndexTraits
 *   is a plain, metric- and dimension-independent alias.
 *
 *   The <Metric, Dim> aliases have NO defaults: downstream code MUST state both,
 *   e.g.
 *       dist_func_t<DistanceMetricsT::EUCLIDEAN, 128> dist;     // no ctor arg
 *       artea_graph::index_t<DistanceMetricsT::COSINE, 112> idx(...);
 *   Metric-independent aliases (vector_dataset_t, distance_t, the compactor, the
 *   compact graph, base scalars, ...) are used without angle brackets as before.
 */

#pragma once

#include <artea/cpu/framework/artea.hpp>

namespace artea {
namespace cpu {

/** ===================================================================== **/
/**  Metric/dim-INDEPENDENT trait chain (BaseTraits / BufferTraits / IndexTraits) **/
/** ===================================================================== **/
using vec_num_t = uint32_t;
using vec_ele_t = float;

using base_traits_t   = BaseTraits<vec_num_t, vec_ele_t>;
using buffer_traits_t = BufferTraits<base_traits_t, BufferPolicyT::LOCKED_BUFFER_WITH_MUTEX, 32>;
using index_traits_t  = IndexTraits<base_traits_t>;

/** ===================================================================== **/
/**  Metric/dim-INDEPENDENT member aliases (no <Metric, Dim>)                  **/
/** ===================================================================== **/

// Base types from BaseTraits
using vec_dim_t = typename base_traits_t::vec_dim_t;
using vertex_num_t = typename base_traits_t::vertex_num_t;
using vertex_id_t = typename base_traits_t::vertex_id_t;
using vec_id_t = typename base_traits_t::vec_id_t;
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
using nbr_comp_t = typename base_traits_t::nbr_comp_t;
using strict_nbr_comp_t = typename base_traits_t::strict_nbr_comp_t;
using nbr_id_comp_t = typename base_traits_t::nbr_id_comp_t;
using nbr_dist_comp_t = typename base_traits_t::nbr_dist_comp_t;
using nbr_arr_t = typename base_traits_t::nbr_arr_t;
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

// Buffer types from BufferTraits
using buffer_policy_t = typename buffer_traits_t::buffer_policy_t;
using log_buffer_t = typename buffer_traits_t::log_buffer_t;
using log_container_t = typename buffer_traits_t::log_container_t;
using log_table_t = typename buffer_traits_t::log_table_t;

// Config / index types from IndexTraits
using layer_config_t = typename index_traits_t::layer_config_t;
using refining_graph_compactor_t = typename index_traits_t::refining_graph_compactor_t;
using flat_graph_file_manager_t = typename index_traits_t::flat_graph_file_manager_t;
using hierarchical_graph_file_manager_t = typename index_traits_t::hierarchical_graph_file_manager_t;
using index_size_calculator_t = typename index_traits_t::index_size_calculator_t;
using radius_prober_t = typename index_traits_t::radius_prober_t;
using hierarchical_graph_compactor_t = typename index_traits_t::hierarchical_graph_compactor_t;

// Graph-storage types from IndexTraits (metric/dim-independent)
namespace compact {
    using refining_graph_t     = typename index_traits_t::compact::refining_graph_t;
    using hierarchical_graph_t = typename index_traits_t::compact::hierarchical_graph_t;
}   // namespace compact
namespace dynamic {
    using hierarchical_graph_t = typename index_traits_t::dynamic::hierarchical_graph_t;
}   // namespace dynamic

// Utility types
using index_register_util_t = IndexRegisterUtil;

/** ===================================================================== **/
/**  Metric/dim-DEPENDENT trait chain (everything below flows from <Metric, Dim>) **/
/** ===================================================================== **/
template <DistanceMetricsT Metric, vec_dim_t Dim>
using computer_traits_t = ComputerTraits<base_traits_t, Metric, Dim>;

template <DistanceMetricsT Metric, vec_dim_t Dim>
using vertex_generator_traits_t = VertexGeneratorTraits<computer_traits_t<Metric, Dim>, index_traits_t>;

template <DistanceMetricsT Metric, vec_dim_t Dim>
using router_traits_t = RouterTraits<computer_traits_t<Metric, Dim>, index_traits_t, false>;

template <DistanceMetricsT Metric, vec_dim_t Dim>
using refiner_traits_t = RefinerTraits<computer_traits_t<Metric, Dim>, buffer_traits_t, index_traits_t, router_traits_t<Metric, Dim>>;

template <DistanceMetricsT Metric, vec_dim_t Dim>
using graph_factory_traits_t = GraphFactoryTraits<
    vertex_generator_traits_t<Metric, Dim>,
    refiner_traits_t<Metric, Dim>,
    router_traits_t<Metric, Dim>
>;

/** ===================================================================== **/
/**  Metric/dim-DEPENDENT member aliases (parameterized by <Metric, Dim>)      **/
/** ===================================================================== **/

// Computer types from ComputerTraits
template <DistanceMetricsT Metric, vec_dim_t Dim> using distance_metrics_t = typename computer_traits_t<Metric, Dim>::distance_metrics_t;
template <DistanceMetricsT Metric, vec_dim_t Dim> using dist_func_t = typename computer_traits_t<Metric, Dim>::dist_func_t;
template <DistanceMetricsT Metric, vec_dim_t Dim> using simdu1_dist_t = typename computer_traits_t<Metric, Dim>::simdu1_dist_t;
template <DistanceMetricsT Metric, vec_dim_t Dim> using simdu2_dist_t = typename computer_traits_t<Metric, Dim>::simdu2_dist_t;
template <DistanceMetricsT Metric, vec_dim_t Dim> using simdu4_dist_t = typename computer_traits_t<Metric, Dim>::simdu4_dist_t;
template <DistanceMetricsT Metric, vec_dim_t Dim> using linear_func_t = typename computer_traits_t<Metric, Dim>::linear_func_t;
template <DistanceMetricsT Metric, vec_dim_t Dim> using simdu1_linear_t = typename computer_traits_t<Metric, Dim>::simdu1_linear_t;
template <DistanceMetricsT Metric, vec_dim_t Dim> using simdu2_linear_t = typename computer_traits_t<Metric, Dim>::simdu2_linear_t;
template <DistanceMetricsT Metric, vec_dim_t Dim> using simdu4_linear_t = typename computer_traits_t<Metric, Dim>::simdu4_linear_t;
template <DistanceMetricsT Metric, vec_dim_t Dim> using recall_estimator_t = typename computer_traits_t<Metric, Dim>::recall_estimator_t;
template <DistanceMetricsT Metric, vec_dim_t Dim> using adr_estimator_t = typename computer_traits_t<Metric, Dim>::adr_estimator_t;
template <DistanceMetricsT Metric, vec_dim_t Dim> using distance_prober_t = typename computer_traits_t<Metric, Dim>::distance_prober_t;
template <DistanceMetricsT Metric, vec_dim_t Dim> using dataset_prober_t = typename computer_traits_t<Metric, Dim>::dataset_prober_t;

// Unified routers from RouterTraits
template <DistanceMetricsT Metric, vec_dim_t Dim> using single_layer_router_t = typename router_traits_t<Metric, Dim>::single_layer_router_t;
template <DistanceMetricsT Metric, vec_dim_t Dim> using hierarchical_graph_router_t = typename router_traits_t<Metric, Dim>::hierarchical_graph_router_t;

// Router data structures / helpers from RouterTraits
template <DistanceMetricsT Metric, vec_dim_t Dim> using candidate_entry_t = typename router_traits_t<Metric, Dim>::candidate_entry_t;
template <DistanceMetricsT Metric, vec_dim_t Dim> using result_entry_t = typename router_traits_t<Metric, Dim>::result_entry_t;
template <DistanceMetricsT Metric, vec_dim_t Dim> using knn_results_t = typename router_traits_t<Metric, Dim>::knn_results_t;
template <DistanceMetricsT Metric, vec_dim_t Dim> using std_candidate_queue_t = typename router_traits_t<Metric, Dim>::std_candidate_queue_t;
template <DistanceMetricsT Metric, vec_dim_t Dim> using linear_candidate_queue_t = typename router_traits_t<Metric, Dim>::linear_candidate_queue_t;
template <DistanceMetricsT Metric, vec_dim_t Dim> using fh_candidate_queue_t = typename router_traits_t<Metric, Dim>::fh_candidate_queue_t;
template <DistanceMetricsT Metric, vec_dim_t Dim> using boost_candidate_queue_t = typename router_traits_t<Metric, Dim>::boost_candidate_queue_t;
template <DistanceMetricsT Metric, vec_dim_t Dim> using candidate_queue_t = typename router_traits_t<Metric, Dim>::candidate_queue_t;
template <DistanceMetricsT Metric, vec_dim_t Dim> using visited_table_pool_t = typename router_traits_t<Metric, Dim>::visited_table_pool_t;
template <DistanceMetricsT Metric, vec_dim_t Dim> using candidate_sample_utils_t = typename router_traits_t<Metric, Dim>::candidate_sample_utils_t;
template <DistanceMetricsT Metric, vec_dim_t Dim> using bruteforce_router_t = typename router_traits_t<Metric, Dim>::bruteforce_router_t;
template <DistanceMetricsT Metric, vec_dim_t Dim> using sl_router_profiler_t = typename router_traits_t<Metric, Dim>::sl_router_profiler_t;
template <DistanceMetricsT Metric, vec_dim_t Dim> using hg_router_profiler_t = typename router_traits_t<Metric, Dim>::hg_router_profiler_t;

// Refiner types from RefinerTraits
template <DistanceMetricsT Metric, vec_dim_t Dim> using triangle_updater_t = typename refiner_traits_t<Metric, Dim>::triangle_updater_t;
template <DistanceMetricsT Metric, vec_dim_t Dim> using hierarchical_pruning_updater_t = typename refiner_traits_t<Metric, Dim>::hierarchical_pruning_updater_t;
template <DistanceMetricsT Metric, vec_dim_t Dim> using reverse_updater_t = typename refiner_traits_t<Metric, Dim>::reverse_updater_t;
template <DistanceMetricsT Metric, vec_dim_t Dim> using random_updater_t = typename refiner_traits_t<Metric, Dim>::random_updater_t;
template <DistanceMetricsT Metric, vec_dim_t Dim> using routing_updater_t = typename refiner_traits_t<Metric, Dim>::routing_updater_t;
template <DistanceMetricsT Metric, vec_dim_t Dim> using truncate_updater_t = typename refiner_traits_t<Metric, Dim>::truncate_updater_t;
template <DistanceMetricsT Metric, vec_dim_t Dim> using random_eg_t = typename refiner_traits_t<Metric, Dim>::random_eg_t;
template <DistanceMetricsT Metric, vec_dim_t Dim> using ivf_partitions_t = typename refiner_traits_t<Metric, Dim>::ivf_partitions_t;
template <DistanceMetricsT Metric, vec_dim_t Dim> using ivf_construct_policy_t = typename refiner_traits_t<Metric, Dim>::ivf_construct_policy_t;
template <DistanceMetricsT Metric, vec_dim_t Dim> using refiner_utils_t = typename refiner_traits_t<Metric, Dim>::refiner_utils_t;
template <DistanceMetricsT Metric, vec_dim_t Dim> using propagate_engine_t = typename refiner_traits_t<Metric, Dim>::propagate_engine_t;

// Vertex generator types from VertexGeneratorTraits
template <DistanceMetricsT Metric, vec_dim_t Dim> using approx_rnet_t = typename vertex_generator_traits_t<Metric, Dim>::approx_rnet_t;
template <DistanceMetricsT Metric, vec_dim_t Dim> using ortho_lsh_generator_t = typename vertex_generator_traits_t<Metric, Dim>::ortho_lsh_generator_t;
template <DistanceMetricsT Metric, vec_dim_t Dim> using pstable_lsh_generator_t = typename vertex_generator_traits_t<Metric, Dim>::pstable_lsh_generator_t;
template <DistanceMetricsT Metric, vec_dim_t Dim> using lsh_table_t = typename vertex_generator_traits_t<Metric, Dim>::lsh_table_t;
template <DistanceMetricsT Metric, vec_dim_t Dim> using lb_greedy_vg_t = typename vertex_generator_traits_t<Metric, Dim>::lb_greedy_vg_t;
template <DistanceMetricsT Metric, vec_dim_t Dim> using mb_greedy_vg_t = typename vertex_generator_traits_t<Metric, Dim>::mb_greedy_vg_t;
template <DistanceMetricsT Metric, vec_dim_t Dim> using random_vg_t = typename vertex_generator_traits_t<Metric, Dim>::random_vg_t;
template <DistanceMetricsT Metric, vec_dim_t Dim> using graph_mis_vg_t = typename vertex_generator_traits_t<Metric, Dim>::graph_mis_vg_t;

// Namespace-scoped graph factories / indexes / configs from GraphFactoryTraits
namespace conv_graph {
    template <DistanceMetricsT Metric, vec_dim_t Dim> using index_t = typename graph_factory_traits_t<Metric, Dim>::conv_graph::index_t;
    template <DistanceMetricsT Metric, vec_dim_t Dim> using factory_t = typename graph_factory_traits_t<Metric, Dim>::conv_graph::factory_t;
    template <DistanceMetricsT Metric, vec_dim_t Dim> using propagate_config_t = typename graph_factory_traits_t<Metric, Dim>::conv_graph::propagate_config_t;
    template <DistanceMetricsT Metric, vec_dim_t Dim> using pruning_config_t = typename graph_factory_traits_t<Metric, Dim>::conv_graph::pruning_config_t;
}   // namespace conv_graph

namespace knn_graph {
    template <DistanceMetricsT Metric, vec_dim_t Dim> using index_t = typename graph_factory_traits_t<Metric, Dim>::knn_graph::index_t;
    template <DistanceMetricsT Metric, vec_dim_t Dim> using factory_t = typename graph_factory_traits_t<Metric, Dim>::knn_graph::factory_t;
    template <DistanceMetricsT Metric, vec_dim_t Dim> using propagate_config_t = typename graph_factory_traits_t<Metric, Dim>::knn_graph::propagate_config_t;
}   // namespace knn_graph

namespace symmetric_knn_graph {
    template <DistanceMetricsT Metric, vec_dim_t Dim> using index_t = typename graph_factory_traits_t<Metric, Dim>::symmetric_knn_graph::index_t;
    template <DistanceMetricsT Metric, vec_dim_t Dim> using factory_t = typename graph_factory_traits_t<Metric, Dim>::symmetric_knn_graph::factory_t;
    template <DistanceMetricsT Metric, vec_dim_t Dim> using propagate_config_t = typename graph_factory_traits_t<Metric, Dim>::symmetric_knn_graph::propagate_config_t;
}   // namespace symmetric_knn_graph

namespace stacked_rgraph {
    template <DistanceMetricsT Metric, vec_dim_t Dim> using rgraph_config_t = typename graph_factory_traits_t<Metric, Dim>::stacked_rgraph::rgraph_config_t;
    template <DistanceMetricsT Metric, vec_dim_t Dim> using pruning_config_t = typename graph_factory_traits_t<Metric, Dim>::stacked_rgraph::pruning_config_t;
    template <DistanceMetricsT Metric, vec_dim_t Dim> using index_t = typename graph_factory_traits_t<Metric, Dim>::stacked_rgraph::index_t;
    template <DistanceMetricsT Metric, vec_dim_t Dim> using factory_t = typename graph_factory_traits_t<Metric, Dim>::stacked_rgraph::factory_t;
}   // namespace stacked_rgraph

namespace artea_graph {
    template <DistanceMetricsT Metric, vec_dim_t Dim> using rgraph_config_t = typename graph_factory_traits_t<Metric, Dim>::artea_graph::rgraph_config_t;
    template <DistanceMetricsT Metric, vec_dim_t Dim> using propagate_config_t = typename graph_factory_traits_t<Metric, Dim>::artea_graph::propagate_config_t;
    template <DistanceMetricsT Metric, vec_dim_t Dim> using pruning_config_t = typename graph_factory_traits_t<Metric, Dim>::artea_graph::pruning_config_t;
    template <DistanceMetricsT Metric, vec_dim_t Dim> using index_t = typename graph_factory_traits_t<Metric, Dim>::artea_graph::index_t;
    template <DistanceMetricsT Metric, vec_dim_t Dim> using factory_t = typename graph_factory_traits_t<Metric, Dim>::artea_graph::factory_t;
}   // namespace artea_graph

namespace hier_conv_graph {
    template <DistanceMetricsT Metric, vec_dim_t Dim> using hierarchy_config_t = typename graph_factory_traits_t<Metric, Dim>::hier_conv_graph::hierarchy_config_t;
    template <DistanceMetricsT Metric, vec_dim_t Dim> using propagate_config_t = typename graph_factory_traits_t<Metric, Dim>::hier_conv_graph::propagate_config_t;
    template <DistanceMetricsT Metric, vec_dim_t Dim> using pruning_config_t = typename graph_factory_traits_t<Metric, Dim>::hier_conv_graph::pruning_config_t;
    template <DistanceMetricsT Metric, vec_dim_t Dim> using index_t = typename graph_factory_traits_t<Metric, Dim>::hier_conv_graph::index_t;
    template <DistanceMetricsT Metric, vec_dim_t Dim> using factory_t = typename graph_factory_traits_t<Metric, Dim>::hier_conv_graph::factory_t;
}   // namespace hier_conv_graph

}   // namespace cpu
}   // namespace artea
