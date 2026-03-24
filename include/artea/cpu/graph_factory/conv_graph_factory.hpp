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
 * @FilePath: /Artea/include/artea/cpu/graph_factory/conv_graph_factory.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2026-02-11 09:56:36
 * @Date: 2025-11-15 20:36:29
 * @Description:
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <chrono>
#include <fmt/format.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/spin_mutex.h>

#include <artea/common/logger.hpp>

namespace artea {
namespace cpu {

template <typename GraphFactoryTraitsT>
class ConvGraphFactory :
    public GraphFactoryTraitsT::template flat_graph_factory_t<ConvGraphFactory<GraphFactoryTraitsT>>
{

    using vertex_num_t = typename GraphFactoryTraitsT::vertex_num_t;
    using vertex_id_t = typename GraphFactoryTraitsT::vertex_id_t;
    using vec_ele_t = typename GraphFactoryTraitsT::vec_ele_t;
    using iter_t = typename GraphFactoryTraitsT::iter_t;
    using ratio_t = typename GraphFactoryTraitsT::ratio_t;
    using flat_graph_t = typename GraphFactoryTraitsT::flat_graph_t;
    using vector_dataset_t = typename GraphFactoryTraitsT::vector_dataset_t;
    using dist_func_t = typename GraphFactoryTraitsT::dist_func_t;
    using layer_config_t = typename GraphFactoryTraitsT::layer_config_t;
    using edges_builder_config_t = typename GraphFactoryTraitsT::conv_graph::edges_builder_config_t;
    // Using propagate_engine_t with no selective scheduling currently.
    using random_eg_t = typename GraphFactoryTraitsT::random_eg_t;
    using propagate_engine_t = typename GraphFactoryTraitsT::template propagate_engine_t<false>;
    using triangle_updater_t = typename GraphFactoryTraitsT::triangle_updater_t;
    using reverse_updater_t = typename GraphFactoryTraitsT::reverse_updater_t;
    using routing_updater_t = typename GraphFactoryTraitsT::routing_updater_t;
    using vector_array_t = typename GraphFactoryTraitsT::vector_array_t;
    using query_vecs_t = typename GraphFactoryTraitsT::query_vecs_t;
    using ground_truth_t = typename GraphFactoryTraitsT::ground_truth_t;
    using recall_estimator_t = typename GraphFactoryTraitsT::recall_estimator_t;
    using graph_mode_t = typename GraphFactoryTraitsT::graph_mode_t;
    using monolayer_graph_router_t = typename GraphFactoryTraitsT::template monolayer_graph_router_t<graph_mode_t::construct_mode>;

public:
    ConvGraphFactory() {}

    /** @brief construct a new convergent graph from vector array */
    auto construct_graph_impl(
        const vector_array_t& base_vecs,
        layer_config_t layer_config,
        edges_builder_config_t edges_builder_config
    ) -> flat_graph_t {
        const vertex_num_t num_vertices = static_cast<vertex_num_t>(base_vecs.get_num_vecs());
        flat_graph_t flat_graph(
            /* vecs_data =          */ base_vecs,
            /* layer_config =       */ layer_config,
            /* edges_builder_config = */ edges_builder_config
        );
        dist_func_t dist_func(base_vecs.get_vec_dim());

        // generate random edges first
        random_eg_t random_eg(dist_func);
        const vertex_num_t init_nbr_size = static_cast<vertex_num_t>(
            layer_config.max_nbr_size() * edges_builder_config.prefill_ratio()
        );
        random_eg.generate(flat_graph, /* init_nbr_size = */ init_nbr_size);
        propagate_engine_t propagate_engine(num_vertices, dist_func);
        propagate_engine.set_graph(flat_graph);
        // Create triangle updater and reverse updater
        auto triangle_updater = propagate_engine.template make_updater<triangle_updater_t>(
            edges_builder_config.scale_coeffs(),
            edges_builder_config.shifted_coeffs()
        );
        auto reverse_updater = propagate_engine.template make_updater<reverse_updater_t>();
        auto routing_updater = propagate_engine.template make_updater<routing_updater_t>(init_nbr_size, init_nbr_size);
        // run propagation engine to refine the graph
        for (iter_t outer_iter = 0; outer_iter < edges_builder_config.num_outer_iters(); ++outer_iter) {
            // propagate_engine.run(1, reverse_updater);
            propagate_engine.run(edges_builder_config.num_inner_iters(), triangle_updater);
            // if (outer_iter != edges_builder_config.num_outer_iters() - 1) {
            //     propagate_engine.run(1, reverse_updater);
            // }
            propagate_engine.run(1, routing_updater);
            propagate_engine.run(1, reverse_updater);
            if (outer_iter == edges_builder_config.num_outer_iters() - 1) {
                propagate_engine.run(1, triangle_updater, false);
            }
        }

        return flat_graph;
    }

    /** @brief construct a new convergent graph from dataset, with per-outer-iter recall/throughput profiling */
    auto construct_graph_impl(
        const vector_dataset_t& dataset,
        layer_config_t layer_config,
        edges_builder_config_t edges_builder_config
    ) -> flat_graph_t {
        const vector_array_t& base_vecs = dataset.get_base_vecs();
        const query_vecs_t& query_vecs = dataset.get_query_vecs();
        const ground_truth_t& groundtruth = dataset.get_gt_vecs();
        const vertex_num_t num_vertices = static_cast<vertex_num_t>(base_vecs.get_num_vecs());

        flat_graph_t flat_graph(base_vecs, layer_config, edges_builder_config);
        dist_func_t dist_func(base_vecs.get_vec_dim());

        random_eg_t random_eg(dist_func);
        const vertex_num_t init_nbr_size = static_cast<vertex_num_t>(
            layer_config.max_nbr_size() * edges_builder_config.prefill_ratio()
        );
        random_eg.generate(flat_graph, init_nbr_size);
        propagate_engine_t propagate_engine(num_vertices, dist_func);
        propagate_engine.set_graph(flat_graph);
        auto triangle_updater = propagate_engine.template make_updater<triangle_updater_t>(
            edges_builder_config.scale_coeffs(),
            edges_builder_config.shifted_coeffs()
        );
        auto reverse_updater = propagate_engine.template make_updater<reverse_updater_t>();
        auto routing_updater = propagate_engine.template make_updater<routing_updater_t>(init_nbr_size, init_nbr_size);

        recall_estimator_t recall_estimator;
        const vertex_num_t topk = groundtruth.get_vec_dim();
        monolayer_graph_router_t router(base_vecs, dist_func, flat_graph, topk, topk);
        router.initialize();

        for (iter_t outer_iter = 0; outer_iter < edges_builder_config.num_outer_iters(); ++outer_iter) {
            propagate_engine.run(edges_builder_config.num_inner_iters(), triangle_updater);
            propagate_engine.run(1, reverse_updater);
            propagate_engine.run(1, routing_updater);
            if (outer_iter == edges_builder_config.num_outer_iters() - 1) {
                propagate_engine.run(1, triangle_updater, false);
            }

            auto t0 = std::chrono::high_resolution_clock::now();
            auto results = router.batch_query(query_vecs);
            auto t1 = std::chrono::high_resolution_clock::now();
            double qps = query_vecs.get_num_vecs() * 1e6 /
                std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
            double recall = recall_estimator.calculate_recall_at_k(results, groundtruth, topk, query_vecs.get_num_vecs());

            ARTEA_INFO(fmt::format(
                "OuterIter {}: Recall@{}={:.4f}, QPS={:.2f}",
                outer_iter, topk, recall, qps
            ));
        }

        return flat_graph;
    }

};  // class ConvGraphFactory


}   // namespace cpu
}   // namespace artea
