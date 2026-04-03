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
 * @FilePath: /Artea/include/artea/cpu/vertex_generator/graph_mis_vg.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Graph-based MIS vertex generator for R-net construction.
 *               Uses a 3-phase (Propose/Veto/Commit) algorithm to handle
 *               asymmetric edges in approximate KNN graphs.
 */

#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <vector>
#include <random>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <artea/common/logger.hpp>
#include <fmt/format.h>

namespace artea {
namespace cpu {

/**
 * @brief Graph-based MIS (Maximal Independent Set) vertex generator.
 *
 * Produces an R-net from a pre-built KNN graph by running a parallel MIS
 * algorithm on the "close neighbor" subgraph (neighbors with distance < min_radius).
 *
 * Uses a 3-phase approach per round to handle asymmetric edges:
 *   - Phase 1 (Propose): Each vertex tentatively decides IN/OUT/UNDECIDED.
 *   - Phase 2 (Veto): IN vertices veto conflicting IN neighbors along out-edges.
 *   - Phase 3a (Commit & Crush): Non-vetoed IN vertices commit and suppress neighbors.
 *   - Phase 3b (Frontier Rebuild): Collect remaining UNDECIDED vertices.
 *
 * @tparam VertexGeneratorTraitsT The vertex generator traits type.
 */
template <typename VertexGeneratorTraitsT>
class GraphMISVG : public VertexGeneratorTraitsT::template vertex_generator_t<GraphMISVG<VertexGeneratorTraitsT>> {

    using vec_num_t = typename VertexGeneratorTraitsT::vec_num_t;
    using vec_id_t = typename VertexGeneratorTraitsT::vec_id_t;
    using vertex_num_t = typename VertexGeneratorTraitsT::vertex_num_t;
    using vertex_id_t = typename VertexGeneratorTraitsT::vertex_id_t;
    using distance_t = typename VertexGeneratorTraitsT::distance_t;
    using vec_dim_t = typename VertexGeneratorTraitsT::vec_dim_t;
    using vector_array_t = typename VertexGeneratorTraitsT::vector_array_t;
    using nbr_arr_t = typename VertexGeneratorTraitsT::nbr_arr_t;
    using approx_rnet_t = typename VertexGeneratorTraitsT::approx_rnet_t;
    using knn_graph = typename VertexGeneratorTraitsT::knn_graph;

    static constexpr uint8_t UNDECIDED = 0;
    static constexpr uint8_t IN = 1;
    static constexpr uint8_t OUT = 2;

public:
    GraphMISVG() = default;

    /**
     * @brief Generate an R-net from a KNN graph using coarse-to-fine parallel 3-phase MIS.
     *
     * When max_power=0, behaves as a single-shot MIS at rnet_radius.
     * When max_power>0, starts MIS at rnet_radius * radix^max_power and
     * divides by radix each step, locking in well-separated seeds at
     * coarse scales before filling gaps at finer scales.
     *
     * Radius schedule: r * radix^max_power, r * radix^(max_power-1), ..., r * radix, r
     *
     * @param graph The KNN graph (knn_graph::index_t, neighbors sorted by distance ascending).
     * @param rnet_radius The final target radius threshold.
     * @param radix Geometric ratio between consecutive radius steps (must be > 1).
     * @param max_power Number of coarse steps above rnet_radius (0 = single-shot).
     * @return approx_rnet_t containing the selected vertex IDs and their vector data.
     */
    auto generate_impl(
        const typename knn_graph::index_t& graph,
        const distance_t rnet_radius,
        const distance_t radix = distance_t(2),
        const uint32_t max_power = uint32_t(0)
    ) -> approx_rnet_t {
        const vertex_num_t num_vertices = graph.get_num_vertices();
        const vector_array_t& vecs_data = graph.get_vecs_data();

        // Build radius schedule: [r * radix^max_power, ..., r * radix, r]
        std::vector<distance_t> radius_schedule;
        {
            for (int32_t p = static_cast<int32_t>(max_power); p >= 1; --p) {
                radius_schedule.push_back(rnet_radius * std::pow(radix, p));
            }
            radius_schedule.push_back(rnet_radius);
        }

        #ifdef ARTEA_PROFILING
        ARTEA_INFO(fmt::format("GraphMISVG: {} radius steps, from {:.6f} to {:.6f}",
            radius_schedule.size(), radius_schedule.front(), radius_schedule.back()));
        #endif

        // Assign random priorities (higher = wins tie-breaking)
        std::vector<uint32_t> priority(num_vertices);
        {
            std::mt19937 rng(42);
            for (vertex_num_t i = 0; i < num_vertices; ++i) {
                priority[i] = rng();
            }
        }

        // Persistent state array across all radius steps
        std::vector<std::atomic<uint8_t>> state(num_vertices);
        for (vertex_num_t i = 0; i < num_vertices; ++i) {
            state[i].store(UNDECIDED, std::memory_order_relaxed);
        }

        // Per-round buffers (reused across steps)
        std::vector<uint8_t> proposal(num_vertices, UNDECIDED);
        std::vector<std::atomic<bool>> veto(num_vertices);
        for (vertex_num_t i = 0; i < num_vertices; ++i) {
            veto[i].store(false, std::memory_order_relaxed);
        }

        // TLS frontier buffer — lives across all steps and MIS rounds to reuse capacity
        tbb::enumerable_thread_specific<std::vector<vertex_id_t>> tls_frontier;

        auto _merge_tls_frontier = [&]() {
            size_t total = 0;
            for (const auto& local : tls_frontier) { total += local.size(); }
            std::vector<vertex_id_t> result;
            result.reserve(total);
            for (auto& local : tls_frontier) {
                result.insert(result.end(), local.begin(), local.end());
                local.clear();
            }
            return result;
        };

        uint32_t total_rounds = 0;

        for (size_t step = 0; step < radius_schedule.size(); ++step) {
            const distance_t current_radius = radius_schedule[step];

            // Reset OUT -> UNDECIDED (IN vertices persist)
            if (step > 0) {
                tbb::parallel_for(
                    tbb::blocked_range<vertex_num_t>(0, num_vertices),
                    [&](const tbb::blocked_range<vertex_num_t>& r) {
                        for (vertex_num_t v = r.begin(); v != r.end(); ++v) {
                            if (state[v].load(std::memory_order_relaxed) == OUT) {
                                state[v].store(UNDECIDED, std::memory_order_relaxed);
                            }
                        }
                    }
                );

                // Pre-Crush: existing IN vertices suppress close neighbors
                // at the new (smaller) radius. This handles asymmetric edges
                // where an IN vertex v sees u (v→u) but u does not see v.
                // Without this, u would remain UNDECIDED, propose IN, and
                // no one would veto it — breaking independence.
                tbb::parallel_for(
                    tbb::blocked_range<vertex_num_t>(0, num_vertices),
                    [&](const tbb::blocked_range<vertex_num_t>& r) {
                        for (vertex_num_t v = r.begin(); v != r.end(); ++v) {
                            if (state[v].load(std::memory_order_relaxed) != IN) continue;
                            const auto& nbrs = graph.fetch_nbrs(v);
                            for (vertex_num_t i = 0; i < nbrs.size(); ++i) {
                                if (nbrs[i].get_distance() >= current_radius) break;
                                state[nbrs[i].get_id()].store(OUT, std::memory_order_relaxed);
                            }
                        }
                    }
                );
            }

            // Build frontier from UNDECIDED vertices (parallel)
            tbb::parallel_for(
                tbb::blocked_range<vertex_num_t>(0, num_vertices),
                [&](const tbb::blocked_range<vertex_num_t>& r) {
                    auto& local = tls_frontier.local();
                    for (vertex_num_t v = r.begin(); v != r.end(); ++v) {
                        if (state[v].load(std::memory_order_relaxed) == UNDECIDED) {
                            local.push_back(v);
                        }
                    }
                }
            );
            std::vector<vertex_id_t> frontier = _merge_tls_frontier();

            uint32_t step_rounds = 0;
            while (!frontier.empty()) {

                // Phase 0: Reset per-round buffers
                tbb::parallel_for(
                    tbb::blocked_range<size_t>(0, frontier.size()),
                    [&](const tbb::blocked_range<size_t>& r) {
                        for (size_t fi = r.begin(); fi != r.end(); ++fi) {
                            const vertex_id_t v = frontier[fi];
                            proposal[v] = UNDECIDED;
                            veto[v].store(false, std::memory_order_relaxed);
                        }
                    }
                );

                // Phase 1: Propose
                tbb::parallel_for(
                    tbb::blocked_range<size_t>(0, frontier.size()),
                    [&](const tbb::blocked_range<size_t>& r) {
                        for (size_t fi = r.begin(); fi != r.end(); ++fi) {
                            const vertex_id_t v = frontier[fi];
                            const auto& nbrs = graph.fetch_nbrs(v);

                            bool any_nbr_in = false;
                            bool has_higher_priority_undecided = false;

                            for (vertex_num_t i = 0; i < nbrs.size(); ++i) {
                                if (nbrs[i].get_distance() >= current_radius) break;

                                const vertex_id_t u = nbrs[i].get_id();
                                const uint8_t u_state = state[u].load(std::memory_order_relaxed);

                                if (u_state == IN) {
                                    any_nbr_in = true;
                                    break;
                                }
                                if (u_state == UNDECIDED &&
                                    (priority[u] > priority[v] ||
                                     (priority[u] == priority[v] && u > v))) {
                                    has_higher_priority_undecided = true;
                                }
                            }

                            if (any_nbr_in) {
                                proposal[v] = OUT;
                            } else if (!has_higher_priority_undecided) {
                                proposal[v] = IN;
                            } else {
                                proposal[v] = UNDECIDED;
                            }
                        }
                    }
                );

                // Phase 2: Veto
                tbb::parallel_for(
                    tbb::blocked_range<size_t>(0, frontier.size()),
                    [&](const tbb::blocked_range<size_t>& r) {
                        for (size_t fi = r.begin(); fi != r.end(); ++fi) {
                            const vertex_id_t v = frontier[fi];
                            if (proposal[v] != IN) continue;

                            const auto& nbrs = graph.fetch_nbrs(v);
                            for (vertex_num_t i = 0; i < nbrs.size(); ++i) {
                                if (nbrs[i].get_distance() >= current_radius) break;

                                const vertex_id_t u = nbrs[i].get_id();
                                if (proposal[u] == IN) {
                                    veto[u].store(true, std::memory_order_relaxed);
                                }
                            }
                        }
                    }
                );

                // Phase 3a: Commit & Crush
                tbb::parallel_for(
                    tbb::blocked_range<size_t>(0, frontier.size()),
                    [&](const tbb::blocked_range<size_t>& r) {
                        for (size_t fi = r.begin(); fi != r.end(); ++fi) {
                            const vertex_id_t v = frontier[fi];

                            if (proposal[v] == OUT) {
                                state[v].store(OUT, std::memory_order_relaxed);
                            } else if (proposal[v] == IN && !veto[v].load(std::memory_order_relaxed)) {
                                state[v].store(IN, std::memory_order_relaxed);
                                const auto& nbrs = graph.fetch_nbrs(v);
                                for (vertex_num_t i = 0; i < nbrs.size(); ++i) {
                                    if (nbrs[i].get_distance() >= current_radius) break;
                                    state[nbrs[i].get_id()].store(OUT, std::memory_order_relaxed);
                                }
                            }
                        }
                    }
                );

                // Phase 3b: Build next frontier (reuse TLS across rounds)
                tbb::parallel_for(
                    tbb::blocked_range<size_t>(0, frontier.size()),
                    [&](const tbb::blocked_range<size_t>& r) {
                        auto& local = tls_frontier.local();
                        for (size_t fi = r.begin(); fi != r.end(); ++fi) {
                            const vertex_id_t v = frontier[fi];
                            if (state[v].load(std::memory_order_relaxed) == UNDECIDED) {
                                local.push_back(v);
                            }
                        }
                    }
                );

                step_rounds++;
                frontier = _merge_tls_frontier();
            }

            total_rounds += step_rounds;

            #ifdef ARTEA_PROFILING
            // Count current IN vertices
            uint32_t count_in = 0;
            for (vertex_num_t v = 0; v < num_vertices; ++v) {
                if (state[v].load(std::memory_order_relaxed) == IN) count_in++;
            }
            ARTEA_INFO(fmt::format("C2F step {}/{}: radius={:.6f}, {} MIS rounds, {} IN vertices",
                step + 1, radius_schedule.size(), current_radius, step_rounds, count_in));
            #endif
        }

        // Collect IN vertices
        approx_rnet_t result(vecs_data.get_vec_dim());
        for (vertex_id_t v = 0; v < num_vertices; ++v) {
            if (state[v].load(std::memory_order_relaxed) == IN) {
                result.vec_ids.push_back(v);
            }
        }
        result.vecs_data = vecs_data.extract_subset(result.vec_ids);

        ARTEA_INFO(fmt::format("GraphMISVG: selected {} / {} vertices ({:.2f}%) in {} rounds",
            result.vec_ids.size(), num_vertices,
            100.0 * result.vec_ids.size() / num_vertices, total_rounds));

        return result;
    }

};  // class GraphMISVG

}   // namespace cpu
}   // namespace artea
