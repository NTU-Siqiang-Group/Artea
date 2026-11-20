// Copyright 2026 Weitang Ye
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <utility>

#include <tbb/parallel_for.h>

#include <artea/cpu/index/practheo/vertex_weight_assigner.hpp>
#include <artea/cpu/router/bruteforce_router.hpp>

namespace artea::cpu::practheo {

/** @brief Construction interface for the Practheo flat graph. */
template <typename GraphFactoryTraitsT>
class IndexFactory {
    using this_index_t = typename GraphFactoryTraitsT::practheo::index_t;
    using vector_array_t = typename GraphFactoryTraitsT::vector_array_t;
    using vertex_id_t = typename GraphFactoryTraitsT::vertex_id_t;
    using vertex_num_t = typename GraphFactoryTraitsT::vertex_num_t;
    using ratio_t = typename GraphFactoryTraitsT::ratio_t;
    using dist_func_t = typename GraphFactoryTraitsT::dist_func_t;
    using bruteforce_router_t = typename GraphFactoryTraitsT::bruteforce_router_t;
    using knn_results_t = typename GraphFactoryTraitsT::knn_results_t;
    using nbr_arr_t = typename GraphFactoryTraitsT::nbr_arr_t;
    using strict_nbr_comp_t = typename GraphFactoryTraitsT::strict_nbr_comp_t;
    using sl_config_t = typename GraphFactoryTraitsT::sl_config_t;
    using build_config_t = typename GraphFactoryTraitsT::practheo::build_config_t;
    using vertex_weight_assigner_t = typename GraphFactoryTraitsT::practheo::vertex_weight_assigner_t;

public:
    /** @brief The constructed index and its end-to-end construction time in milliseconds. */
    struct ConstructResult {
        this_index_t graph;
        double total_time_ms = 0.0;
    };

    /** @brief Assign weights to a batch of base vectors starting at global VID first_vid. */
    static auto assign_vertex_weight(this_index_t& index, const vector_array_t& batch_vecs,
                                     vertex_id_t first_vid) -> void {
        vertex_weight_assigner_t::assign_vertex_weight(index, batch_vecs, first_vid);
    }

    /** @brief Build outgoing edges by sampling global and per-weight nearest neighbors.
     * The base vectors must have the factory's padded dimension and outlive the returned index.
     */
    template <bool bruteforce = true>
    static auto construct_graph(
        const vector_array_t& base_vecs,
        sl_config_t layer_config,
        build_config_t build_config
    ) -> ConstructResult {
        const auto start = std::chrono::steady_clock::now();
        this_index_t graph(base_vecs, layer_config, build_config);
        assign_vertex_weight(graph, base_vecs, 0);
        if constexpr (bruteforce) {
            bruteforce_build_graph(graph);
        } else {
            // Reserved for approximate neighbor search.
        }
        const std::chrono::duration<double, std::milli> elapsed = std::chrono::steady_clock::now() - start;
        return {std::move(graph), elapsed.count()};
    }

    template <bool bruteforce = true>
    static auto construct_graph(const vector_array_t&&, sl_config_t, build_config_t) -> ConstructResult = delete;

private:
    static auto bruteforce_build_graph(this_index_t& graph) -> void {
        const auto& config = graph.build_config();
        config.validate_sampling_rates();
        const auto& base_vecs = graph.get_base_vecs();
        const auto count = base_vecs.get_num_vecs();
        const auto k = config.num_nearest_nbrs;
        if (count < 2 || k == 0 || (config.sampling_rate0 == 0 && config.sampling_rate1 == 0)) return;
        if (base_vecs.get_vec_dim() != GraphFactoryTraitsT::vec_dim) {
            throw std::invalid_argument("Practheo base vector dimension differs from the distance kernel");
        }

        // One extra hit leaves room to remove the query VID in either search.
        const dist_func_t distance;
        const bruteforce_router_t router(base_vecs, distance, k < count ? k + 1 : count);
        tbb::parallel_for(vertex_id_t{0}, count, [&](vertex_id_t vid) {
            auto& nbrs = graph.fetch_nbrs(vid);
            sample_from_global(graph, router, vid, nbrs);
            sample_from_weight_groups(graph, router, vid, nbrs);

            std::sort(nbrs.begin(), nbrs.end(), strict_nbr_comp_t{});
            nbrs.erase(std::unique(nbrs.begin(), nbrs.end(), [](const auto& a, const auto& b) {
                return a.get_vid() == b.get_vid();
            }), nbrs.end());
        });

        // Compaction uses the layer capacity as the upper bound on the extracted degree.
        auto max_degree = graph.layer_config().max_nbr_size();
        for (const auto& nbrs : graph.get_nbrs_arr()) {
            max_degree = std::max(max_degree, static_cast<vertex_num_t>(nbrs.size()));
        }
        graph.layer_config().max_nbr_size(max_degree);
    }

    static auto nearest_others(knn_results_t hits, vertex_id_t vid, vertex_num_t k) -> knn_results_t {
        std::erase_if(hits, [vid](const auto& hit) { return hit.get_vid() == vid; });
        hits.resize(std::min<std::size_t>(k, hits.size()));
        return hits;
    }

    // Even and odd seeds give each VID separate global and weight-group sampling streams.
    static auto bernoulli_sample(const knn_results_t& candidates, ratio_t rate,
                                 std::uint64_t seed, nbr_arr_t& nbrs) -> void {
        std::mt19937_64 rng(seed);
        std::bernoulli_distribution keep(std::min(rate, ratio_t{1}));
        for (const auto& candidate : candidates) {
            if (keep(rng)) nbrs.emplace_back(candidate.get_vid(), candidate.get_distance(), true);
        }
    }

    static auto sample_from_global(const this_index_t& graph, const bruteforce_router_t& router,
                                   vertex_id_t vid, nbr_arr_t& nbrs) -> void {
        const auto& config = graph.build_config();
        if (config.sampling_rate0 == 0) return;
        const auto nearest = nearest_others(router.query(graph.get_base_vecs().get(vid)), vid, config.num_nearest_nbrs);
        bernoulli_sample(nearest, config.sampling_rate0, 2ULL * vid, nbrs);
    }

    static auto sample_from_weight_groups(const this_index_t& graph, const bruteforce_router_t& router,
                                          vertex_id_t vid, nbr_arr_t& nbrs) -> void {
        const auto& config = graph.build_config();
        if (config.sampling_rate1 == 0) return;
        const auto* query = graph.get_base_vecs().get(vid);
        knn_results_t candidates;
        for (const auto& weight_group : graph.get_vids_by_weight()) {
            const auto nearest = nearest_others(router.query(query, weight_group), vid, config.num_nearest_nbrs);
            candidates.insert(candidates.end(), nearest.begin(), nearest.end());
        }
        // Each VID belongs to exactly one group, so these candidates are already distinct.
        bernoulli_sample(candidates, config.sampling_rate1, 2ULL * vid + 1, nbrs);
    }
};

}  // namespace artea::cpu::practheo
