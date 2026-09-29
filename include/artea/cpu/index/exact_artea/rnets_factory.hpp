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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <artea/cpu/index/exact_artea/index_structure.hpp>

namespace artea {
namespace cpu {
namespace exact_artea {

/** @brief Exact hierarchical r-net vertex-set construction. */
template <typename GraphFactoryTraitsT>
class RNetsFactory {
    using this_index_t = typename GraphFactoryTraitsT::exact_artea::index_t;
    using vertex_id_t = typename GraphFactoryTraitsT::vertex_id_t;
    using vertex_num_t = typename GraphFactoryTraitsT::vertex_num_t;
    using layer_id_t = typename GraphFactoryTraitsT::layer_id_t;
    using distance_t = typename GraphFactoryTraitsT::distance_t;
    using dist_func_t = typename GraphFactoryTraitsT::dist_func_t;
    using hierarchical_graph_t = typename GraphFactoryTraitsT::dynamic::hierarchical_graph_t;

public:
    /**
     * @brief Build exact nested r-net vertex sets by farthest point sampling.
     *
     * V_0 contains every base vector, including duplicates. For h >= 1,
     * V_h is an R_h-net of V_{h-1}, using rgraph_config.radius_at(h):
     * every point of V_{h-1} is within distance <= R_h of V_h, and distinct
     * points of V_h have distance > R_h. Stop at a singleton top layer;
     * empty and singleton datasets have no upper layers. The storage bound
     * follows the actual height, not the stacked-rgraph insertion heuristic.
     * Only vertex membership and empty neighbor slots are constructed.
     *
     * Each layer starts at vertex 0. Repeatedly select the point with the
     * largest exact distance to the selected set, breaking ties by smallest
     * vertex ID. TBB parallelizes distance updates and the argmax reduction.
     * Covered points can be skipped permanently: their nearest distance can
     * only decrease. There is no approximate search, subsampling or epsilon.
     * Membership is deterministic for a deterministic distance functor;
     * physical slot / bucket order may differ between parallel runs.
     *
     * @param index Prepared index over immutable base vectors, with an empty
     *              building graph. Call prepare_build before rebuilding.
     * @param dist_func Thread-safe metric distance functor; radii must use
     *                  the same units (e.g. squared radii for squared L2).
     *                  Exactness is with respect to its returned distances.
     * @throws std::logic_error If the building graph is absent or nonempty.
     * @throws std::domain_error If an evaluated distance is negative/nonfinite.
     * @throws std::overflow_error If a required radius or height is unrepresentable.
     *
     * On failure the index is unchanged. On success previous graph references
     * are invalidated. No concurrent queries or modifications are allowed.
     * Auxiliary sampling space is O(|V_0|); distance work is at most
     * O(sum_h |V_{h-1}| * |V_h|) distance evaluations.
     */
    static auto fps_generator(this_index_t& index, const dist_func_t& dist_func) -> void {
        if (index.get_num_vertices() != 0) {
            throw std::logic_error("fps_generator requires an empty building graph; call prepare_build");
        }
        const auto& vectors = index.get_base_vecs();
        const auto& config = index.rgraph_config();
        const auto count = static_cast<vertex_num_t>(vectors.get_num_vecs());
        constexpr auto invalid_vid = GraphFactoryTraitsT::invalid_vertex_id;
        constexpr auto invalid_level = hierarchical_graph_t::invalid_level_id;
        constexpr std::size_t grain_size = 256;

        std::vector<layer_id_t> highest_level(count, layer_id_t(0));
        std::vector<vertex_id_t> candidates(count);
        std::iota(candidates.begin(), candidates.end(), vertex_id_t(0));
        std::vector<vertex_id_t> selected;
        std::vector<distance_t> nearest_distance;
        layer_id_t h = 0;

        struct Farthest {
            distance_t distance;
            vertex_id_t vid;
            std::size_t position;
        };
        const auto farther = [](const Farthest& a, const Farthest& b) {
            if (a.distance != b.distance) return a.distance > b.distance ? a : b;
            return a.vid < b.vid ? a : b;
        };

        while (candidates.size() > 1) {
            if (h == invalid_level - 1) {
                throw std::overflow_error("Exact r-net height is not representable");
            }
            ++h;
            const distance_t radius = config.radius_at(h);
            if (!std::isfinite(radius) || radius <= distance_t(0)) {
                throw std::overflow_error("Exact r-nets require finite positive layer radii");
            }
            selected.clear();
            nearest_distance.resize(candidates.size());
            std::size_t center_position = 0;

            for (;;) {
                const auto center = candidates[center_position];
                selected.push_back(center);
                highest_level[center] = h;
                nearest_distance[center_position] = distance_t(0);
                const auto* center_vector = vectors.get(center);
                const bool first_center = selected.size() == 1;

                const auto farthest = tbb::parallel_reduce(
                    tbb::blocked_range<std::size_t>(0, candidates.size(), grain_size),
                    Farthest{radius, invalid_vid, 0},
                    [&](const tbb::blocked_range<std::size_t>& range, Farthest best) {
                        for (std::size_t i = range.begin(); i != range.end(); ++i) {
                            if (i == center_position ||
                                (!first_center && nearest_distance[i] <= radius)) continue;
                            const auto vid = candidates[i];
                            const distance_t distance = dist_func(vectors.get(vid), center_vector);
                            if (!std::isfinite(distance) || distance < distance_t(0)) {
                                throw std::domain_error("Exact r-nets require finite nonnegative distances");
                            }
                            const auto nearest = first_center ? distance :
                                std::min(nearest_distance[i], distance);
                            nearest_distance[i] = nearest;
                            if (nearest > radius) best = farther(best, Farthest{nearest, vid, i});
                        }
                        return best;
                    }, farther);
                if (farthest.vid == invalid_vid) break;
                center_position = farthest.position;
            }
            candidates.swap(selected);
        }

        // assign_layer may only be called once per vertex. Finish sampling first,
        // then allocate each vertex's complete set of layers in a private graph.
        auto graph = std::make_unique<hierarchical_graph_t>(
            h, config.ul_max_nbr_size(), config.bl_max_nbr_size(), count);
        graph->add_vertices(count);
        tbb::parallel_for(tbb::blocked_range<std::size_t>(0, count, grain_size),
            [&](const tbb::blocked_range<std::size_t>& range) {
                for (std::size_t i = range.begin(); i != range.end(); ++i) {
                    graph->assign_layer(static_cast<vertex_id_t>(i), highest_level[i]);
                }
            });
        index.replace_building_graph(std::move(graph));
    }
};

}   // namespace exact_artea
}   // namespace cpu
}   // namespace artea
