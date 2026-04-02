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
 */

#pragma once

#include <algorithm>
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
 * R-net guarantee (on the close-neighbor subgraph):
 *   - Independence: no two selected vertices are close neighbors (distance < min_radius).
 *   - Maximality: every non-selected vertex has at least one selected close neighbor.
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
     * @brief Generate an R-net from a KNN graph using parallel MIS.
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

        // Double-buffered state arrays
        std::vector<uint8_t> state(num_vertices, UNDECIDED);
        std::vector<uint8_t> new_state(num_vertices, UNDECIDED);

        // Frontier: vertices still UNDECIDED
        std::vector<vertex_id_t> frontier(num_vertices);
        std::iota(frontier.begin(), frontier.end(), 0);

        uint32_t round = 0;
        while (!frontier.empty()) {
            // Phase 1: Tentative marking (read from state, write to new_state)
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
                            if (state[u] == IN) {
                                any_nbr_in = true;
                                break;
                            }
                            if (state[u] == UNDECIDED &&
                                (priority[u] > priority[v] ||
                                 (priority[u] == priority[v] && u > v))) {
                                has_higher_priority_undecided = true;
                            }
                        }

                        if (any_nbr_in) {
                            new_state[v] = OUT;
                        } else if (!has_higher_priority_undecided) {
                            new_state[v] = IN;
                        } else {
                            new_state[v] = UNDECIDED;
                        }
                    }
                }
            );

            // Phase 2: Commit state and build next frontier
            std::vector<vertex_id_t> next_frontier;
            next_frontier.reserve(frontier.size());

            for (const auto& v : frontier) {
                state[v] = new_state[v];
                if (state[v] == UNDECIDED) {
                    next_frontier.push_back(v);
                }
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
            if (state[v] == IN) {
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
