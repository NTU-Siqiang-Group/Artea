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
 * @FilePath: /Artea/include/artea/cpu/framework/artea.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Including the file is all you need.
 */

#pragma once

/** @note: this file is only provided for project user,
 *         do not include this file in internal implementation. */

#include <artea/common/logger.hpp>

#include <artea/cpu/framework/type_traits/base_traits.hpp>
#include <artea/cpu/framework/type_traits/computer_traits.hpp>
#include <artea/cpu/framework/type_traits/buffer_traits.hpp>
#include <artea/cpu/framework/type_traits/router_traits.hpp>
#include <artea/cpu/framework/type_traits/vertex_generator_traits.hpp>
#include <artea/cpu/framework/type_traits/refiner_traits.hpp>
#include <artea/cpu/framework/type_traits/index_traits.hpp>
#include <artea/cpu/framework/type_traits/graph_factory_traits.hpp>

#include <artea/cpu/containers/allocator.hpp>
#include <artea/cpu/containers/vector_array.hpp>
#include <artea/cpu/containers/vector_dataset.hpp>
#include <artea/cpu/containers/locked_buffer.hpp>
#include <artea/cpu/containers/tbb_buffer.hpp>
#include <artea/cpu/containers/word_aligned_bitmap.hpp>
#include <artea/cpu/containers/vertex_subset.hpp>
#include <artea/cpu/containers/four_ary_heap.hpp>

#include <artea/cpu/index/layer_config.hpp>
#include <artea/cpu/index/artea_graph/vertices_builder_config.hpp>
#include <artea/cpu/index/layer_nbr.hpp>
#include <artea/cpu/index/descent_nbr.hpp>
#include <artea/cpu/index/compact_internal_graph.hpp>
#include <artea/cpu/index/internal_graph.hpp>
#include <artea/cpu/index/descent_graph.hpp>
#include <artea/cpu/index/compact_descent_graph.hpp>
#include <artea/cpu/index/inter_layer_links.hpp>
#include <artea/cpu/index/hierarchical_graph.hpp>
#include <artea/cpu/index/hierarchical_graph_v2.hpp>
#include <artea/cpu/index/hierarchical_search_graph.hpp>
#include <artea/cpu/index/hierarchy_manager.hpp>
#include <artea/cpu/index/descent_graph_compactor.hpp>
#include <artea/cpu/index/internal_graph_compactor.hpp>
#include <artea/cpu/index/index_size_calculator.hpp>
#include <artea/cpu/index/persistence/flat_graph_file_manager.hpp>
#include <artea/cpu/index/persistence/hierarchical_graph_file_manager.hpp>

#include <artea/cpu/index/conv_graph/configs.hpp>
#include <artea/cpu/index/conv_graph/index_structure.hpp>
#include <artea/cpu/index/conv_graph/index_factory.hpp>
#include <artea/cpu/index/knn_graph/configs.hpp>
#include <artea/cpu/index/knn_graph/index_structure.hpp>
#include <artea/cpu/index/knn_graph/index_factory.hpp>
#include <artea/cpu/index/symmetric_knn_graph/configs.hpp>
#include <artea/cpu/index/symmetric_knn_graph/index_structure.hpp>
#include <artea/cpu/index/symmetric_knn_graph/index_factory.hpp>
#include <artea/cpu/index/artea_graph/configs.hpp>
#include <artea/cpu/index/artea_graph/index_structure.hpp>
#include <artea/cpu/index/artea_graph/index_factory.hpp>

#include <artea/cpu/router/vector_router.hpp>
#include <artea/cpu/router/bruteforce_router.hpp>
#include <artea/cpu/router/search_mode_monolayer_graph_router.hpp>
#include <artea/cpu/router/search_mode_hierarchical_graph_router.hpp>
#include <artea/cpu/router/construct_mode_monolayer_graph_router.hpp>
#include <artea/cpu/router/construct_mode_hierarchical_graph_router.hpp>
#include <artea/cpu/router/data_structures/candidate_entry.hpp>
#include <artea/cpu/router/candidate_queue_concept.hpp>
#include <artea/cpu/router/visited_table_concept.hpp>
#include <artea/cpu/router/data_structures/visited_table_pool.hpp>
#include <artea/cpu/router/data_structures/std_candidate_queue.hpp>
#include <artea/cpu/router/data_structures/fh_candidate_queue.hpp>
#include <artea/cpu/router/data_structures/linear_candidate_queue.hpp>

#include <artea/cpu/vertex_generator/vertex_generator.hpp>
#include <artea/cpu/vertex_generator/lsh_table.hpp>
#include <artea/cpu/vertex_generator/pstable_lsh_generator.hpp>
#include <artea/cpu/vertex_generator/ortho_lsh_generator.hpp>
#include <artea/cpu/vertex_generator/lb_greedy_vg.hpp>
#include <artea/cpu/vertex_generator/mb_greedy_vg.hpp>
#include <artea/cpu/vertex_generator/random_vg.hpp>
#include <artea/cpu/vertex_generator/graph_mis_vg.hpp>

#include <artea/cpu/refiner/nbr_log_table.hpp>
#include <artea/cpu/refiner/neighbor_updater.hpp>
#include <artea/cpu/refiner/triangle_updater.hpp>
#include <artea/cpu/refiner/pruning_updater.hpp>
#include <artea/cpu/refiner/reverse_updater.hpp>
#include <artea/cpu/refiner/random_updater.hpp>
#include <artea/cpu/refiner/random_eg.hpp>
#include <artea/cpu/refiner/propagate_engine.hpp>
#include <artea/cpu/refiner/ivf_partitions.hpp>
#include <artea/cpu/refiner/routing_updater.hpp>
#include <artea/cpu/refiner/truncate_updater.hpp>

#include <artea/cpu/utils/bit_ops.hpp>
#include <artea/cpu/utils/centroid_computer.hpp>
#include <artea/cpu/utils/clear_cache.hpp>
#include <artea/cpu/utils/nbr_arr_checker.hpp>
#include <artea/cpu/utils/parallel.hpp>
#include <artea/cpu/utils/random_seq.hpp>
#include <artea/cpu/utils/distance_prober.hpp>
#include <artea/cpu/utils/dataset_prober.hpp>
#include <artea/cpu/utils/radius_prober.hpp>
#include <artea/cpu/utils/index_register_util.hpp>
#include <artea/cpu/utils/simd_distance.hpp>
#include <artea/cpu/utils/simd_fma.hpp>
#include <artea/cpu/utils/simd_linear.hpp>
#include <artea/cpu/utils/vector_sampler.hpp>
#include <artea/cpu/utils/recall_estimator.hpp>
