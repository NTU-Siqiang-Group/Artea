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
 * @FilePath: /Artea/include/artea/cpu/index/knn_graph/index_factory.hpp
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
#include <functional>
#include <fmt/format.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/spin_mutex.h>

#include <artea/common/logger.hpp>

namespace artea {
namespace cpu {
namespace knn_graph {

template <typename GraphFactoryTraitsT>
class IndexFactory {

    using vertex_num_t = typename GraphFactoryTraitsT::vertex_num_t;
    using vertex_id_t = typename GraphFactoryTraitsT::vertex_id_t;
    using vec_ele_t = typename GraphFactoryTraitsT::vec_ele_t;
    using iter_t = typename GraphFactoryTraitsT::iter_t;
    using ratio_t = typename GraphFactoryTraitsT::ratio_t;
    using vector_dataset_t = typename GraphFactoryTraitsT::vector_dataset_t;
    using dist_func_t = typename GraphFactoryTraitsT::dist_func_t;
    using layer_config_t = typename GraphFactoryTraitsT::layer_config_t;
    using this_index_t = typename GraphFactoryTraitsT::knn_graph::index_t;
    using propagate_config_t = typename GraphFactoryTraitsT::knn_graph::propagate_config_t;
    using pruning_config_t = typename GraphFactoryTraitsT::knn_graph::pruning_config_t;
    using random_eg_t        = typename GraphFactoryTraitsT::random_eg_t;
    using propagate_engine_t = typename GraphFactoryTraitsT::propagate_engine_t;
    using triangle_updater_t = typename GraphFactoryTraitsT::triangle_updater_t;
    using reverse_updater_t  = typename GraphFactoryTraitsT::reverse_updater_t;
    using routing_updater_t  = typename GraphFactoryTraitsT::routing_updater_t;
    using truncate_updater_t = typename GraphFactoryTraitsT::truncate_updater_t;
    using vector_array_t = typename GraphFactoryTraitsT::vector_array_t;
    using query_vecs_t = typename GraphFactoryTraitsT::query_vecs_t;
    using ground_truth_t = typename GraphFactoryTraitsT::ground_truth_t;
    using recall_estimator_t = typename GraphFactoryTraitsT::recall_estimator_t;
    using refining_graph_router_t = typename GraphFactoryTraitsT::dynamic::refining_graph_router_t;

public:
    /** @brief construct a new KNN graph from vector array */
    static auto construct_graph(
        const vector_array_t& base_vecs,
        const layer_config_t layer_config,
        const propagate_config_t propagate_config
    ) -> this_index_t {
        this_index_t graph_index(base_vecs, layer_config, propagate_config);
        dist_func_t dist_func(base_vecs.get_vec_dim());
        _build_loop(graph_index, dist_func, propagate_config);
        return graph_index;
    }

    /** @brief construct a new KNN graph from dataset, with per-build-loop recall/throughput profiling */
    static auto profile_graph_quality(
        const vector_dataset_t& dataset,
        const layer_config_t layer_config,
        const propagate_config_t propagate_config
    ) -> void {
        const vector_array_t& base_vecs = dataset.get_base_vecs();
        const query_vecs_t& query_vecs = dataset.get_query_vecs();
        const ground_truth_t& groundtruth = dataset.get_gt_vecs();

        this_index_t graph_index(base_vecs, layer_config, propagate_config);
        dist_func_t dist_func(base_vecs.get_vec_dim());

        recall_estimator_t recall_estimator;
        const vertex_num_t topk = 20;
        const vertex_num_t candidate_queue_size = 40;
        refining_graph_router_t router(
            base_vecs, dist_func,
            graph_index.get_refining_graph(),
            topk, candidate_queue_size);
        router.initialize();

        _build_loop(graph_index, dist_func, propagate_config,
            [&](iter_t build_loop) {
                auto t0 = std::chrono::high_resolution_clock::now();
                auto results = router.batch_query(query_vecs);
                auto t1 = std::chrono::high_resolution_clock::now();
                double qps = query_vecs.get_num_vecs() * 1e6 /
                    std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
                double recall = recall_estimator.calculate_recall_at_k(
                    results, groundtruth, topk, query_vecs.get_num_vecs());
                ARTEA_INFO(fmt::format(
                    "BuildLoop {}: Recall@{}={:.4f}, QPS={:.2f}, Candidate={}",
                    build_loop, topk, recall, qps, candidate_queue_size
                ));
            }
        );
    }

private:

    /**
     * @brief Core build loop shared by all construct_graph overloads.
     *
     * Initializes random edges, creates all updaters, then runs the iteration
     * schedule. An optional per-iter callback is invoked at the end of each
     * build loop (e.g. for recall/QPS profiling in the dataset overload).
     *
     * @param graph_index       The graph being constructed (modified in-place).
     * @param dist_func        Distance function for this graph.
     * @param pruning_config   Pruning configuration (scale_coeffs, shifted_coeffs).
     * @param propagate_config Propagation configuration (num_build_loops, num_triangle_updater_iters, prefill_ratio).
     * @param on_iter_end      Optional callback called after each build loop with
     *                         the current build loop index. Pass nullptr to skip.
     */
    static auto _build_loop(
        this_index_t& graph_index,
        const dist_func_t& dist_func,
        const propagate_config_t& propagate_config,
        std::function<void(iter_t)> on_iter_end = nullptr
    ) -> void {
        const vertex_num_t num_vertices = graph_index.get_num_vertices();
        const vertex_num_t max_nbr_size = graph_index.layer_config().max_nbr_size();
        const vertex_num_t init_nbr_size = static_cast<vertex_num_t>(max_nbr_size * propagate_config.prefill_ratio());

        /** -------------------- Optimazation ------------------------------------- ***/
        /** @brief A sparse graph is effecient enough to search nearest neighbors     */
        graph_index.layer_config().max_nbr_size(max_nbr_size / 2);
        /** ----------------------------------------------------------------------- ***/

        random_eg_t random_eg(dist_func);
        random_eg.generate(graph_index.get_refining_graph(), init_nbr_size);

        propagate_engine_t propagate_engine(dist_func);
        propagate_engine.set_graph(graph_index.get_refining_graph());

        const auto& pruning_config = graph_index.pruning_config();
        auto triangle_updater  = propagate_engine.template make_updater<triangle_updater_t>(
            pruning_config.scale_coeffs(), pruning_config.shifted_coeffs());
        auto reverse_updater   = propagate_engine.template make_updater<reverse_updater_t>();
        const vertex_num_t routing_topk = propagate_config.resolve_routing_topk(max_nbr_size);
        const vertex_num_t routing_queue_size = propagate_config.resolve_routing_queue_size(max_nbr_size);
        auto routing_updater   = propagate_engine.template make_updater<routing_updater_t>(routing_topk, routing_queue_size);
        auto truncate_updater  = propagate_engine.template make_updater<truncate_updater_t>();

        for (iter_t build_loop = 0; build_loop < propagate_config.num_build_loops(); ++build_loop) {
            propagate_engine.run(propagate_config.num_triu_iters(), triangle_updater)
                            .next(reverse_updater).next(truncate_updater);
            if (on_iter_end) { on_iter_end(build_loop); }
        }

        /** -------------------- Optimazation --------------------------------------- ***/
        /** @brief Reconstructed as dense graph with original edge number requirements  */
        graph_index.layer_config().max_nbr_size(max_nbr_size);
        /** ------------------------------------------------------------------------- ***/

        for (iter_t routing_loop = 0; routing_loop < propagate_config.num_routing_loops(); ++routing_loop) {
            propagate_engine.next(routing_updater);
            if (on_iter_end) { on_iter_end(propagate_config.num_build_loops() + routing_loop); }
        }
    }

};  // class IndexFactory

}   // namespace knn_graph
}   // namespace cpu
}   // namespace artea
