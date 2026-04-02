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
     * @brief Generate an R-net from a KNN graph using parallel 3-phase MIS.
     *
     * @param graph The KNN graph (knn_graph::index_t, neighbors sorted by distance ascending).
     * @param min_radius The radius threshold: neighbors with distance < min_radius
     *                   are considered "close" and mutually exclusive in the R-net.
     * @return approx_rnet_t containing the selected vertex IDs and their vector data.
     */
    auto generate_impl(
        const typename knn_graph::index_t& graph,
        const distance_t min_radius
    ) -> approx_rnet_t {
        const vertex_num_t num_vertices = graph.get_num_vertices();
        const vector_array_t& vecs_data = graph.get_vecs_data();

        // Assign random priorities (higher = wins tie-breaking)
        std::vector<uint32_t> priority(num_vertices);
        {
            std::mt19937 rng(42);
            for (vertex_num_t i = 0; i < num_vertices; ++i) {
                priority[i] = rng();
            }
        }

        // Atomic state array for lock-free parallel commit & crush
        std::vector<std::atomic<uint8_t>> state(num_vertices);
        for (vertex_num_t i = 0; i < num_vertices; ++i) {
            state[i].store(UNDECIDED, std::memory_order_relaxed);
        }

        // Per-round proposal buffer (each thread writes a distinct index, no contention)
        std::vector<uint8_t> proposal(num_vertices, UNDECIDED);

        // Per-round veto flags (false→true single-direction, relaxed atomics)
        std::vector<std::atomic<bool>> veto(num_vertices);
        for (vertex_num_t i = 0; i < num_vertices; ++i) {
            veto[i].store(false, std::memory_order_relaxed);
        }

        // Frontier: vertices still UNDECIDED
        std::vector<vertex_id_t> frontier(num_vertices);
        std::iota(frontier.begin(), frontier.end(), 0);

        uint32_t round = 0;
        while (!frontier.empty()) {

            // Phase 0: Reset per-round buffers (parallel)
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

            // Phase 1: Propose (parallel)
            // Read state[], write proposal[] (each thread writes distinct indices)
            tbb::parallel_for(
                tbb::blocked_range<size_t>(0, frontier.size()),
                [&](const tbb::blocked_range<size_t>& r) {
                    for (size_t fi = r.begin(); fi != r.end(); ++fi) {
                        const vertex_id_t v = frontier[fi];
                        const auto& nbrs = graph.fetch_nbrs(v);

                        bool any_nbr_in = false;
                        bool has_higher_priority_undecided = false;

                        for (vertex_num_t i = 0; i < nbrs.size(); ++i) {
                            if (nbrs[i].get_distance() >= min_radius) break;

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

            // Phase 2: Veto (parallel)
            // IN proposers veto conflicting IN neighbors along out-edges
            tbb::parallel_for(
                tbb::blocked_range<size_t>(0, frontier.size()),
                [&](const tbb::blocked_range<size_t>& r) {
                    for (size_t fi = r.begin(); fi != r.end(); ++fi) {
                        const vertex_id_t v = frontier[fi];
                        if (proposal[v] != IN) continue;

                        const auto& nbrs = graph.fetch_nbrs(v);
                        for (vertex_num_t i = 0; i < nbrs.size(); ++i) {
                            if (nbrs[i].get_distance() >= min_radius) break;

                            const vertex_id_t u = nbrs[i].get_id();
                            if (proposal[u] == IN) {
                                veto[u].store(true, std::memory_order_relaxed);
                            }
                        }
                    }
                }
            );

            // Phase 3a: Commit & Crush (parallel)
            // Non-vetoed IN vertices commit and suppress close neighbors
            tbb::parallel_for(
                tbb::blocked_range<size_t>(0, frontier.size()),
                [&](const tbb::blocked_range<size_t>& r) {
                    for (size_t fi = r.begin(); fi != r.end(); ++fi) {
                        const vertex_id_t v = frontier[fi];

                        if (proposal[v] == OUT) {
                            state[v].store(OUT, std::memory_order_relaxed);
                        } else if (proposal[v] == IN && !veto[v].load(std::memory_order_relaxed)) {
                            state[v].store(IN, std::memory_order_relaxed);
                            // Crush: suppress all close neighbors along out-edges
                            const auto& nbrs = graph.fetch_nbrs(v);
                            for (vertex_num_t i = 0; i < nbrs.size(); ++i) {
                                if (nbrs[i].get_distance() >= min_radius) break;
                                state[nbrs[i].get_id()].store(OUT, std::memory_order_relaxed);
                            }
                        }
                        // Vetoed IN and UNDECIDED: do not write state, remain UNDECIDED
                    }
                }
            );

            // Phase 3b: Build next frontier (parallel collect + sequential merge)
            tbb::enumerable_thread_specific<std::vector<vertex_id_t>> tls_frontier;

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

            // Pre-compute total size, single allocation, then copy
            size_t total = 0;
            for (const auto& local : tls_frontier) {
                total += local.size();
            }

            std::vector<vertex_id_t> next_frontier;
            next_frontier.reserve(total);
            for (const auto& local : tls_frontier) {
                next_frontier.insert(next_frontier.end(), local.begin(), local.end());
            }

            round++;
            ARTEA_INFO(fmt::format("MIS round {}: frontier {} -> {}, decided {} vertices",
                round, frontier.size(), next_frontier.size(),
                frontier.size() - next_frontier.size()));

            frontier = std::move(next_frontier);
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
            100.0 * result.vec_ids.size() / num_vertices, round));

        return result;
    }

};  // class GraphMISVG

}   // namespace cpu
}   // namespace artea
