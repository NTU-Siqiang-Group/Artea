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

#include <chrono>
#include <artea/common/logger.hpp>
#include <fmt/format.h>

namespace artea {
namespace cpu {
namespace artea_graph {

template <typename GraphFactoryTraitsT>
class IndexFactory {

    using vertex_num_t = typename GraphFactoryTraitsT::vertex_num_t;
    using vertex_id_t = typename GraphFactoryTraitsT::vertex_id_t;
    using layer_id_t = typename GraphFactoryTraitsT::layer_id_t;
    using layer_num_t = typename GraphFactoryTraitsT::layer_num_t;
    using ratio_t = typename GraphFactoryTraitsT::ratio_t;
    using distance_t = typename GraphFactoryTraitsT::distance_t;
    using dist_func_t = typename GraphFactoryTraitsT::dist_func_t;
    using conv_graph  = typename GraphFactoryTraitsT::conv_graph;
    using knn_graph   = typename GraphFactoryTraitsT::knn_graph;
    using artea_graph = typename GraphFactoryTraitsT::artea_graph;
    using this_index_t = typename artea_graph::index_t;
    using vector_array_t = typename GraphFactoryTraitsT::vector_array_t;
    using vertex_subset_t = typename GraphFactoryTraitsT::vertex_subset_t;
    using graph_mis_vg_t = typename GraphFactoryTraitsT::graph_mis_vg_t;
    using radius_prober_t = typename GraphFactoryTraitsT::radius_prober_t;
    using centroid_computer_t = typename GraphFactoryTraitsT::centroid_computer_t;
    using bruteforce_router_t = typename GraphFactoryTraitsT::bruteforce_router_t;
    using layer_config_t = typename GraphFactoryTraitsT::layer_config_t;
    using rnet_config_t = typename artea_graph::rnet_config_t;

    static constexpr vertex_num_t min_num_layer_vertex = GraphFactoryTraitsT::min_num_layer_vertex;

public:

    static auto construct_graph(
        const vector_array_t& base_vecs,
        layer_config_t bottom_layer_config,
        layer_config_t upper_layer_config,
        typename artea_graph::pruning_config_t bottom_pruning_config,
        typename artea_graph::pruning_config_t upper_pruning_config,
        typename artea_graph::propagate_config_t propagate_config,
        rnet_config_t rnet_config
    ) -> this_index_t {
        dist_func_t dist_func(base_vecs.get_vec_dim());

        this_index_t hierarchical_graph(
            base_vecs,
            bottom_layer_config,
            upper_layer_config,
            bottom_pruning_config,
            upper_pruning_config,
            propagate_config,
            rnet_config
        );

        auto& hier_vecs_manager = hierarchical_graph.get_hier_vecs_manager();
        auto& inter_layer_links = hierarchical_graph.get_inter_layer_links();

        const vector_array_t* current_layer_vecs = &base_vecs;
        distance_t rnet_radius = distance_t(0);

        layer_id_t layer_id = 0;
        while (true) {
            #ifdef ARTEA_PROFILING
            auto t0 = std::chrono::high_resolution_clock::now();
            #endif

            // Select configs for this layer
            const auto& layer_config = (layer_id == 0) ? bottom_layer_config : upper_layer_config;
            const auto& pruning_config = (layer_id == 0) ? bottom_pruning_config : upper_pruning_config;

            // Step 1: Build KNN graph
            auto knn_graph_index = knn_graph::factory_t::construct_graph(
                *current_layer_vecs, layer_config,
                static_cast<typename knn_graph::pruning_config_t>(pruning_config),
                static_cast<typename knn_graph::propagate_config_t>(propagate_config));

            #ifdef ARTEA_PROFILING
            auto t1 = std::chrono::high_resolution_clock::now();
            #endif

            // Step 2: Probe rnet_radius (L0 only; L1+ uses prev * rnet_beta)
            if (layer_id == 0) {
                radius_prober_t prober;
                auto probe_result = prober.probe(knn_graph_index, rnet_config_t::target_rank, rnet_config_t::target_quantile);
                rnet_radius = probe_result.radius * rnet_config.rnet_beta();
                ARTEA_INFO(fmt::format("Layer 0: probed rnet_radius={:.4f} (rank={}, quantile={:.3f}, beta={:.2f})",
                    rnet_radius, rnet_config_t::target_rank, rnet_config_t::target_quantile, rnet_config.rnet_beta()));
            }

            // Step 3: Run GraphMIS to select next layer vertices
            graph_mis_vg_t mis_vg;
            vertex_subset_t next_layer_subset = mis_vg.generate(
                knn_graph_index, rnet_radius, rnet_config.mis_radix(), rnet_config.mis_max_power());

            #ifdef ARTEA_PROFILING
            auto t2 = std::chrono::high_resolution_clock::now();
            #endif

            // Step 4: Refine KNN graph to conv_graph
            auto conv_graph_index = conv_graph::factory_t::construct_graph(
                std::move(knn_graph_index),
                static_cast<typename conv_graph::pruning_config_t>(pruning_config));

            // Set layer graph
            hierarchical_graph.resize(layer_id + 1);
            hierarchical_graph.set_layer_graph(layer_id, std::move(conv_graph_index));

            #ifdef ARTEA_PROFILING
            auto t3 = std::chrono::high_resolution_clock::now();
            double knn_time_s = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1e6;
            double mis_time_s = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1e6;
            double refine_time_s = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count() / 1e6;
            ARTEA_INFO(fmt::format("Layer {}: {} vertices, rnet_radius={:.4f}, knn={:.2f}s, mis={:.2f}s, refine={:.2f}s, selected={}",
                layer_id, current_layer_vecs->get_num_vecs(), rnet_radius,
                knn_time_s, mis_time_s, refine_time_s, next_layer_subset.get_num_vecs()));
            #else
            ARTEA_INFO(fmt::format("Layer {}: {} vertices, rnet_radius={:.4f}, selected={}",
                layer_id, current_layer_vecs->get_num_vecs(), rnet_radius, next_layer_subset.get_num_vecs()));
            #endif

            // Step 5: Check termination
            if (next_layer_subset.get_num_vecs() < min_num_layer_vertex) {
                ARTEA_INFO(fmt::format("Stopping: MIS selected {} < min_num_layer_vertex({})",
                    next_layer_subset.get_num_vecs(), min_num_layer_vertex));
                break;
            }

            // Step 6: Append layer and advance
            inter_layer_links.bottom_up_append(std::move(next_layer_subset.vec_ids));
            hier_vecs_manager.bottom_up_append(std::move(next_layer_subset.vecs_data));

            layer_id = hier_vecs_manager.get_num_layers() - 1;
            current_layer_vecs = &hier_vecs_manager.get_layer_vecs(layer_id);
            rnet_radius *= rnet_config.rnet_beta();
        }

        // Set entry point: vertex closest to centroid in top layer
        const layer_id_t top_layer_id = hier_vecs_manager.get_num_layers() - 1;
        const auto& top_layer_vecs = hier_vecs_manager.get_layer_vecs(top_layer_id);
        auto centroid = centroid_computer_t::compute(top_layer_vecs);
        bruteforce_router_t bf_router(top_layer_vecs, dist_func, 1);
        bf_router.initialize();
        auto nearest = bf_router.query(centroid.data());
        hierarchical_graph.set_entry_point(nearest[0].get_id());

        ARTEA_INFO(fmt::format("Artea graph: {} layers, entry_point={} (top layer)",
            hier_vecs_manager.get_num_layers(), nearest[0].get_id()));

        return hierarchical_graph;
    }
};  // class IndexFactory

}   // namespace artea_graph
}   // namespace cpu
}   // namespace artea
