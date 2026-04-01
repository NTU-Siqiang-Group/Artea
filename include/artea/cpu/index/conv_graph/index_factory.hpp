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
 * @FilePath: /Artea/include/artea/cpu/index/conv_graph/index_factory.hpp
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
namespace conv_graph {

template <typename GraphFactoryTraitsT>
class IndexFactory {

    using vertex_num_t = typename GraphFactoryTraitsT::vertex_num_t;
    using vertex_id_t = typename GraphFactoryTraitsT::vertex_id_t;
    using vec_ele_t = typename GraphFactoryTraitsT::vec_ele_t;
    using iter_t = typename GraphFactoryTraitsT::iter_t;
    using ratio_t = typename GraphFactoryTraitsT::ratio_t;
    using this_index_t = typename GraphFactoryTraitsT::conv_graph::index_t;
    using vector_dataset_t = typename GraphFactoryTraitsT::vector_dataset_t;
    using dist_func_t = typename GraphFactoryTraitsT::dist_func_t;
    using layer_config_t = typename GraphFactoryTraitsT::layer_config_t;
    using propagate_config_t = typename GraphFactoryTraitsT::conv_graph::propagate_config_t;
    using pruning_config_t = typename GraphFactoryTraitsT::conv_graph::pruning_config_t;
    // Edge generator types parameterized on this_index_t
    using random_eg_t = typename GraphFactoryTraitsT::random_eg_t;
    using propagate_engine_t = typename GraphFactoryTraitsT::template propagate_engine_t<this_index_t, false>;
    using triangle_updater_t = typename GraphFactoryTraitsT::template triangle_updater_t<this_index_t>;
    using reverse_updater_t = typename GraphFactoryTraitsT::template reverse_updater_t<this_index_t>;
    using routing_updater_t = typename GraphFactoryTraitsT::template routing_updater_t<this_index_t>;
    using truncate_updater_t = typename GraphFactoryTraitsT::template truncate_updater_t<this_index_t>;
    using vector_array_t = typename GraphFactoryTraitsT::vector_array_t;
    using query_vecs_t = typename GraphFactoryTraitsT::query_vecs_t;
    using ground_truth_t = typename GraphFactoryTraitsT::ground_truth_t;
    using recall_estimator_t = typename GraphFactoryTraitsT::recall_estimator_t;
    using graph_mode_t = typename GraphFactoryTraitsT::graph_mode_t;
    using monolayer_graph_router_t = typename GraphFactoryTraitsT::template monolayer_graph_router_t<graph_mode_t::construct_mode>;
    using knn_graph = typename GraphFactoryTraitsT::knn_graph;

public:
    /** @brief construct a new convergent graph from vector array */
    static auto construct_graph(
        const vector_array_t& base_vecs,
        const layer_config_t layer_config,
        const pruning_config_t pruning_config,
        const propagate_config_t propagate_config
    ) -> this_index_t {
        this_index_t flat_graph(base_vecs, layer_config, pruning_config, propagate_config);
        dist_func_t dist_func(base_vecs.get_vec_dim());
        _build_loop(flat_graph, dist_func, pruning_config, propagate_config);
        return flat_graph;
    }

    /**
     * @brief Construct a convergent graph from an existing knn_graph by taking
     *        ownership of its edges, then running triangle+reverse pruning.
     *
     * @warning This function moves from the input knn_graph. After the call,
     *          the input is left in a valid but unspecified state — the caller
     *          must not use it further.
     *
     * @param knn_graph_index  The knn_graph whose edges will be consumed (moved).
     * @param pruning_config   Pruning configuration for triangle updater.
     * @return A fully constructed convergent graph.
     */
    static auto construct_graph(
        typename knn_graph::index_t&& knn_graph_index,
        const pruning_config_t pruning_config
    ) -> this_index_t {
        this_index_t flat_graph(std::move(knn_graph_index));
        flat_graph.pruning_config() = pruning_config;

        const vertex_num_t num_vertices = flat_graph.get_num_vertices();
        dist_func_t dist_func(flat_graph.get_vecs_data().get_vec_dim());

        propagate_engine_t propagate_engine(num_vertices, dist_func);
        propagate_engine.set_graph(flat_graph);

        auto triangle_updater = propagate_engine.template make_updater<triangle_updater_t>(
            pruning_config.scale_coeffs(), pruning_config.shifted_coeffs());
        auto reverse_updater  = propagate_engine.template make_updater<reverse_updater_t>();
        auto truncate_updater = propagate_engine.template make_updater<truncate_updater_t>();

        propagate_engine.next(triangle_updater).next(truncate_updater)
                        .next(reverse_updater).next(truncate_updater);

        return flat_graph;
    }

    /** @brief construct a new convergent graph from dataset, with per-build-loop recall/throughput profiling */
    static auto profile_graph_quality(
        const vector_dataset_t& dataset,
        const layer_config_t layer_config,
        const pruning_config_t pruning_config,
        const propagate_config_t propagate_config
    ) -> void {
        const vector_array_t& base_vecs = dataset.get_base_vecs();
        const query_vecs_t& query_vecs = dataset.get_query_vecs();
        const ground_truth_t& groundtruth = dataset.get_gt_vecs();

        this_index_t flat_graph(base_vecs, layer_config, pruning_config, propagate_config);
        dist_func_t dist_func(base_vecs.get_vec_dim());

        recall_estimator_t recall_estimator;
        const vertex_num_t topk = 20;
        const vertex_num_t candidate_queue_size = 40;
        monolayer_graph_router_t router(base_vecs, dist_func, topk, candidate_queue_size);
        router.initialize();

        _build_loop(flat_graph, dist_func, pruning_config, propagate_config,
            [&](iter_t build_loop) {
                auto t0 = std::chrono::high_resolution_clock::now();
                auto results = router.batch_query(query_vecs, flat_graph);
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
     * @param flat_graph       The graph being constructed (modified in-place).
     * @param dist_func        Distance function for this graph.
     * @param pruning_config   Pruning configuration (scale_coeffs, shifted_coeffs).
     * @param propagate_config Propagation configuration (num_build_loops, num_triangle_updater_iters, prefill_ratio).
     * @param on_iter_end      Optional callback called after each build loop with
     *                         the current build loop index. Pass nullptr to skip.
     */
    static auto _build_loop(
        this_index_t& flat_graph,
        const dist_func_t& dist_func,
        const pruning_config_t& pruning_config,
        const propagate_config_t& propagate_config,
        std::function<void(iter_t)> on_iter_end = nullptr
    ) -> void {
        const vertex_num_t num_vertices = flat_graph.get_num_vertices();
        const vertex_num_t max_nbr_size = flat_graph.layer_config().max_nbr_size();
        const vertex_num_t init_nbr_size = static_cast<vertex_num_t>(max_nbr_size * propagate_config.prefill_ratio());

        random_eg_t random_eg(dist_func);
        random_eg.generate(flat_graph, init_nbr_size);

        propagate_engine_t propagate_engine(num_vertices, dist_func);
        propagate_engine.set_graph(flat_graph);

        auto triangle_updater  = propagate_engine.template make_updater<triangle_updater_t>(
            pruning_config.scale_coeffs(), pruning_config.shifted_coeffs());
        auto reverse_updater   = propagate_engine.template make_updater<reverse_updater_t>();
        auto routing_updater   = propagate_engine.template make_updater<routing_updater_t>(init_nbr_size, init_nbr_size * 2);
        auto truncate_updater  = propagate_engine.template make_updater<truncate_updater_t>();

        for (iter_t build_loop = 0; build_loop < propagate_config.num_build_loops(); ++build_loop) {
            propagate_engine.run(propagate_config.num_triu_iters(), triangle_updater)
                            .next(reverse_updater).next(truncate_updater);
            if (on_iter_end) { on_iter_end(build_loop); }
        }

        for (iter_t routing_loop = 0; routing_loop < propagate_config.num_routing_loops(); ++routing_loop) {
            propagate_engine.next(routing_updater).next(truncate_updater)
                            .next(triangle_updater).next(truncate_updater)
                            .next(reverse_updater).next(truncate_updater);
            if (on_iter_end) { on_iter_end(propagate_config.num_build_loops() + routing_loop); }
        }
    }

};  // class IndexFactory

}   // namespace conv_graph
}   // namespace cpu
}   // namespace artea
